// Inside the Game - Azure Container Apps deployment.
//
// Provisions:
//   * Log Analytics + Container Apps environment
//   * Azure Container Registry (azd pushes the image here)
//   * User-assigned managed identity (pull from ACR, call the model)
//   * Microsoft Foundry resource (AIServices) + a gpt-4.1-mini deployment
//   * Azure Files share mounted at /app/data/generated so generated matches persist
//   * Two container apps from the same image: api (FastAPI/SSE) and ui (Streamlit)
//
// Deploy with `azd up` (see azure.yaml) or
//   az deployment group create -g <rg> -f infra/main.bicep -p environmentName=dev
targetScope = 'resourceGroup'

@minLength(1)
@maxLength(32)
@description('Name of the environment, used to derive resource names')
param environmentName string

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Region for the Foundry model deployment (gpt-4.1-mini availability varies)')
param modelLocation string = 'swedencentral'

@description('Model to deploy')
param modelName string = 'gpt-4.1-mini'
param modelVersion string = '2025-04-14'
@description('Tokens-per-minute capacity in thousands for the model deployment')
param modelCapacity int = 30

@description('Container image for both roles (set by azd; defaults to a placeholder until the first deploy)')
param apiImage string = 'mcr.microsoft.com/k8se/quickstart:latest'
param uiImage string = apiImage

var prefix = toLower('sisy-${environmentName}')
var token = toLower(uniqueString(subscription().id, resourceGroup().id, environmentName))
var tags = { 'azd-env-name': environmentName, project: 'inside-the-game' }

// ---------------------------------------------------------------- observability
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-appi'
  location: location
  tags: tags
  kind: 'web'
  properties: { Application_Type: 'web', WorkspaceResourceId: logs.id }
}

// ---------------------------------------------------------------- identity + registry
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-id'
  location: location
  tags: tags
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'sisy${token}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

var acrPullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, acrPullRole)
  scope: acr
  properties: { roleDefinitionId: acrPullRole, principalId: identity.properties.principalId, principalType: 'ServicePrincipal' }
}

// ---------------------------------------------------------------- Microsoft Foundry
resource foundry 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' = {
  name: '${prefix}-foundry'
  location: modelLocation
  tags: tags
  kind: 'AIServices'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }
  properties: {
    customSubDomainName: '${prefix}-foundry-${token}'
    publicNetworkAccess: 'Enabled'
    allowProjectManagement: true
    disableLocalAuth: true
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  parent: foundry
  name: 'inside-the-game'
  location: modelLocation
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: { displayName: 'Inside the Game', description: 'Match insight agents' }
}

resource model 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: modelName
  sku: { name: 'GlobalStandard', capacity: modelCapacity }
  properties: {
    model: { format: 'OpenAI', name: modelName, version: modelVersion }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

// "Azure AI User" lets the app identity call the project's models with Entra ID.
var aiUserRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '53ca6127-db72-4b80-b1b0-d745d6d5456d')
resource aiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(foundry.id, identity.id, aiUserRole)
  scope: foundry
  properties: { roleDefinitionId: aiUserRole, principalId: identity.properties.principalId, principalType: 'ServicePrincipal' }
}

// ---------------------------------------------------------------- persistent match storage
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'sisy${token}'
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: { allowBlobPublicAccess: false, minimumTlsVersion: 'TLS1_2' }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource share 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: 'matches'
  properties: { shareQuota: 20 }
}

// ---------------------------------------------------------------- container apps
resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-cae'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}

resource envStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: env
  name: 'matches'
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: share.name
      accessMode: 'ReadWrite'
    }
  }
}

var commonEnv = [
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: insights.properties.ConnectionString }
  { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
  { name: 'SISYPHUS_DATA_DIR', value: '/app/data/generated' }
]

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  tags: union(tags, { 'azd-service-name': 'api' })
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'http', allowInsecure: false }
      registries: [ { server: acr.properties.loginServer, identity: identity.id } ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          command: [ 'sisyphus', 'serve', '--port', '8000' ]
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: concat(commonEnv, [
            { name: 'FOUNDRY_PROJECT_ENDPOINT', value: 'https://${foundry.properties.customSubDomainName}.services.ai.azure.com/api/projects/${project.name}' }
            { name: 'FOUNDRY_MODEL', value: model.name }
          ])
          volumeMounts: [ { volumeName: 'matches', mountPath: '/app/data/generated' } ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, initialDelaySeconds: 15, periodSeconds: 30 }
          ]
        }
      ]
      volumes: [ { name: 'matches', storageType: 'AzureFile', storageName: envStorage.name } ]
      scale: { minReplicas: 1, maxReplicas: 2 }
    }
  }
  dependsOn: [ acrPull, aiUser ]
}

resource ui 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-ui'
  location: location
  tags: union(tags, { 'azd-service-name': 'ui' })
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8501, transport: 'http' }
      registries: [ { server: acr.properties.loginServer, identity: identity.id } ]
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: uiImage
          command: [ 'sisyphus', 'ui', '--port', '8501' ]
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(commonEnv, [
            { name: 'SISYPHUS_API_URL', value: 'https://${api.properties.configuration.ingress.fqdn}' }
            { name: 'HEALTH_URL', value: 'http://localhost:8501/_stcore/health' }
          ])
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 2 }
    }
  }
  dependsOn: [ acrPull ]
}

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = acr.properties.loginServer
output API_URL string = 'https://${api.properties.configuration.ingress.fqdn}'
output UI_URL string = 'https://${ui.properties.configuration.ingress.fqdn}'
output FOUNDRY_PROJECT_ENDPOINT string = 'https://${foundry.properties.customSubDomainName}.services.ai.azure.com/api/projects/${project.name}'
output FOUNDRY_MODEL string = model.name
