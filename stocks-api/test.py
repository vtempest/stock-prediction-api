import yfinance as yf
import matplotlib.pyplot as plt

# Download past 2 months of AAPL data with 2-minute intervals
aapl = yf.Ticker("AAPL")
data = aapl.history(period="2mo", interval="2m")

# For daily data within 2 months:
# data = aapl.history(period="2mo")

# Plot the closing price
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], label='AAPL Close')
plt.title('AAPL Closing Price - Past 2 Months (2-min Interval)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Save to PNG file
plt.savefig("aapl_2mo_2m_close.png", dpi=300, bbox_inches='tight')

plt.show()
