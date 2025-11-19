# Overview

## Financial Analysis Agent (Proof-of-Concept)

This project implements a proof-of-concept **financial analysis agent** using **LangGraph** and **LangChain**, designed as a precursor to a future **multi-agent financial analysis and automated trading system**. The agent can process user queries about financial data and return concise, actionable summaries.

### Graph Workflow
- **main node** – Uses an LLM to parse and understand the user’s query, determining the type of financial information requested.  
- **fetcher node** – Retrieves relevant data from APIs (currently **Yahoo Finance** and **Alpha Vantage**).  
- **query node** – Processes the fetched data and performs any calculations or extractions required.  
- **summarizer node** – Condenses the results into a concise summary suitable for user consumption.  

Each node contributes to a streamlined pipeline: the user input is interpreted, data is retrieved and queried, and results are summarized for clear, actionable output.

### Financial Data Sources
- **Yahoo Finance (yfinance)** – Historical and real-time market data.  
- **Alpha Vantage** – Additional stock and financial metrics for enhanced analysis.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/saanviibrahim45/langchain-practice.git
cd langchain-practice
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file and define your API keys as follows:
`OPENAI_API_KEY=your_openai_key`
`ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key`


## 5. Running the agent
```bash
python3 src/main_agent.py
```

