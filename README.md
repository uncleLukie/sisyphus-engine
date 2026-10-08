# Project Title
Sisyphus Engine

# Tagline
Sisyphus Engine tracks physical fatigue and pressing drop-off in real time using synthetic player tracking data, generating live momentum metrics and synchronized broadcast commentary.

# Keywords
Sports Analytics, Player Tracking, Multi-Agent Systems, Real-Time Analytics, Synthetic Telemetry, Microsoft Foundry, Azure Container Apps, Broadcast Analytics, Google Research Football, SPADL

# Description

## Overview
High-pressing teams frequently lose control when physical fatigue sets in, but traditional broadcast statistics only show overall distance or possession share long after the shift occurs. Sisyphus Engine is a real-time analytics pipeline that processes synthetic player tracking data to measure physical drop-off as it happens on the pitch. By calculating rolling sprint volume, average movement speed, and team compactness, the engine detects when a team's press is breaking down and generates live, synchronized commentary explaining tactical momentum shifts.

## Problem
Standard match statistics are static and retrospective. Broadcasters and viewers rarely have visibility into physical exhaustion until a defensive error occurs or a goal is conceded. Sisyphus Engine solves this by tracking physical output continuously across rolling windows, giving commentators and fans immediate insight into energy loss and changing match dynamics.

## Key Features
- Live Fatigue Tracking: Calculates rolling 5-minute averages of player speed, sprint frequency, and defensive team compactness (spatial area covered).
- Team Energy Index: Produces a live momentum score that flags significant drops in pressing intensity.
- Multi-Agent Commentary: Uses specialized agents to identify fatigue triggers and generate context-aware match commentary.
- Dual Viewer Views: Delivers tailored visual outputs, including a simple momentum gauge for casual fans and detailed speed and spatial metrics for tactical analysts.

## Pipeline Architecture
1. Ingest: Reads synthetic tracking data and action events from Google Research Football and SPADL streams using kloppy and pandas.
2. Interpret: Runs continuous sliding windows to calculate speed decay, sprint volume, and team pitch coverage.
3. Explain: Detects fatigue breakpoints and triggers language model agents to explain why a team's defensive structure is failing.
4. Render: Formats analytical payloads into timed JSON ready for broadcast graphics overlays.
5. Personalize: Adjusts data depth and commentary tone for casual viewers versus tactical analysts.

## Tech Stack
- AI and Orchestration: Microsoft Foundry, Azure Agent Framework, GitHub Copilot SDK
- Data Processing: Python 3.12, pandas, numpy, scipy, kloppy, SPADL
- Cloud and Infrastructure: Azure Container Apps, Azure Functions
- Frontend: Streamlit dashboard with pitch visualizer and live commentary feed
- Data Source: Google Research Football synthetic tracking feeds

## Target Categories
- Grand Prize: Complete real-time match intelligence pipeline.
- Best Multi-Agent System: Coordinated multi-agent workflow split across telemetry monitoring, tactical analysis, and audience formatting.
- Best Azure Cloud Native Integration: Scalable, containerized streaming backend deployed on Azure Container Apps.
