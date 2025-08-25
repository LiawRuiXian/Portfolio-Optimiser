# Portfolio Optimiser Wizard

**Portfolio Optimiser Wizard** is an intuitive and interactive web based dashboard that helps **non-finance professionals** to create and optimise their investment portfolios using **Modern Portfolio Theory (MPT)**. . Users enter the stocks they follow and choose a optimisation objective such as **maximising Sharpe ratio, minimising volatility, or minimising volatility for a target return**, etc. The dashboard visualises portfolio performance against a benchmark and calculates key metrics, including **annualised return, Sharpe ratio, maximum drawdown**, etc.
Users can also explore the **efficient frontier**, by hovering over points to see each portfolio’s return, volatility, and asset weights. With this tool, users can easily identify their **optimal portfolio allocation** effortlessly, without needing advanced financial knowledge.
 
## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/portfolio-optimiser-dashboard.git
   cd portfolio-optimiser-dashboard
2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    pip install -r requirements.txt
3. Run the dashboard with Streamlit:
    ```bash
    streamlit run app.py
## Usage
After launching, the dashboard will open in your browser. You can configure the following options:
1. Enter the stock tickers you follow.
2. Select the date range for historical data.
3. Choose the return frequency.
4. Set the DCA (Dollar-Cost Averaging) interval.
5. Select an optimisation objective.

Once configured, you can view:
- Portfolio optimisation results
- Backtested performance
- The efficient frontier  

Explore your portfolio and have fun!

## Technologies Used
- **Python** – Core programming language
- **Streamlit** – Building the interactive web app
- **Pandas & NumPy** – Data manipulation and analysis
- **Plotly** – Interactive visualisations
- **yfinance** – Fetching stock price data

## Future Enhancements
- Add multi-factor portfolio analysis
- Integrate automatic rebalancing recommendations
