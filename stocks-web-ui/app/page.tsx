"use client"
import { useState, useEffect, useCallback } from "react"
import type React from "react"
import dynamic from "next/dynamic"
import type { PlotData, Layout } from "plotly.js"

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

interface QuoteData {
  symbol: string
  longName?: string
  shortName?: string
  regularMarketPrice?: number
  regularMarketChange?: number
  regularMarketChangePercent?: number
  regularMarketVolume?: number
  marketCap?: number
  sector?: string
  industry?: string
}

interface HistoricalData {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface PERatioPoint {
  date: string
  price: number
  ttmEPS: number | null
  peRatio: number | null
}

interface PEStats {
  count: number
  min: number
  max: number
  average: number
  median: number
  current: number | null
}

export default function Home() {
  const [searchSymbol, setSearchSymbol] = useState("")
  const [selectedSymbol, setSelectedSymbol] = useState("AAPL")
  const [period, setPeriod] = useState("1y")

  const [quoteData, setQuoteData] = useState<QuoteData | null>(null)
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([])
  const [peData, setPeData] = useState<PERatioPoint[]>([])
  const [peStats, setPeStats] = useState<PEStats | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [errors, setErrors] = useState<{ source: string; reason: string }[]>([])

  // New state for chart types
  const [priceChartType, setPriceChartType] = useState<'candlestick' | 'line' | 'area'>("candlestick")
  const [volumeChartType, setVolumeChartType] = useState<'bar' | 'line' | 'area'>("bar")

  const fetchStockData = useCallback(async (symbol: string, fetchPeriod: string) => {
    setIsLoading(true)
    setErrors([])

    // Reset data on new fetch
    setQuoteData(null)
    setHistoricalData([])
    setPeData([])
    setPeStats(null)

    try {
      const response = await fetch(`/api/stock-analysis/${symbol}?period=${fetchPeriod}`)
      if (!response.ok) {
        const errorBody = await response.json()
        throw new Error(errorBody.error || "Failed to fetch stock analysis data")
      }

      const result = await response.json()

      if (result.quote) setQuoteData(result.quote)
      if (result.historical)
        setHistoricalData(result.historical.map((d: any) => ({ ...d, date: new Date(d.date).toISOString() })))
      if (result.pe) {
        setPeData(result.pe.peRatios.map((d: any) => ({ ...d, date: new Date(d.date).toISOString() })))
        setPeStats(result.pe.statistics)
      }
      if (result.errors && result.errors.length > 0) {
        setErrors(result.errors)
      }
    } catch (err) {
      setErrors([{ source: "Client", reason: err instanceof Error ? err.message : "A client-side error occurred" }])
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStockData(selectedSymbol, period)
  }, [selectedSymbol, period, fetchStockData])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchSymbol.trim()) {
      setSelectedSymbol(searchSymbol.trim().toUpperCase())
    }
  }

  // Pass chart type to getChartData
  const chartData = getChartData(historicalData, selectedSymbol, priceChartType, volumeChartType)
  const peChartData = getPeChartData(peData, selectedSymbol)

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Stock Analysis Dashboard</h1>
          <div className="text-sm text-gray-500">Backend: Next.js API + yahoo-finance2</div>
        </div>

        {/* Search Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <form onSubmit={handleSearch} className="flex gap-4 items-end">
            <div className="flex-1">
              <label htmlFor="symbol" className="block text-sm font-medium text-gray-700 mb-2">
                Stock Symbol
              </label>
              <input
                id="symbol"
                type="text"
                value={searchSymbol}
                onChange={(e) => setSearchSymbol(e.target.value)}
                placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT)"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label htmlFor="period" className="block text-sm font-medium text-gray-700 mb-2">
                Price Chart Period
              </label>
              <select
                id="period"
                value={period}
                onChange={(e) => setPeriod(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="1mo">1 Month</option>
                <option value="3mo">3 Months</option>
                <option value="6mo">6 Months</option>
                <option value="1y">1 Year</option>
                <option value="2y">2 Years</option>
                <option value="5y">5 Years</option>
              </select>
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Loading..." : "Search"}
            </button>
          </form>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Fetching stock data...</p>
          </div>
        )}

        {/* Error State */}
        {errors.length > 0 && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-red-800 mb-2">
              {errors.length === 3 ? "Failed to Load Data" : "Partial Data Loaded"}
            </h3>
            <p className="text-red-700 mb-2">
              There was an issue fetching some of the stock data. The available data is displayed below.
            </p>
            <ul className="list-disc list-inside text-sm text-red-600 space-y-1">
              {errors.map((err, i) => (
                <li key={i}>
                  <span className="font-semibold capitalize">{err.source} data:</span> {err.reason}
                </li>
              ))}
            </ul>
            <button
              onClick={() => fetchStockData(selectedSymbol, period)}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {/* Data Display */}
        {!isLoading && (quoteData || historicalData.length > 0) && (
          <>
            {/* Stock Info Section */}
            {quoteData && (
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-gray-900">{quoteData.symbol}</h3>
                    <p className="text-sm text-gray-600">{quoteData.longName || quoteData.shortName}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-gray-900">
                      ${quoteData.regularMarketPrice?.toFixed(2) || "N/A"}
                    </p>
                    <p
                      className={`text-sm ${(quoteData.regularMarketChange || 0) >= 0 ? "text-green-600" : "text-red-600"}`}
                    >
                      {quoteData.regularMarketChange?.toFixed(2) || "N/A"} (
                      {quoteData.regularMarketChangePercent?.toFixed(2) || "N/A"}%)
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Volume</p>
                    <p className="text-lg font-semibold">{quoteData.regularMarketVolume?.toLocaleString() || "N/A"}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Market Cap</p>
                    <p className="text-lg font-semibold">
                      {quoteData.marketCap ? formatMarketCap(quoteData.marketCap) : "N/A"}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* P/E Ratio Stats */}
            {peStats && (
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Historical P/E Ratio Statistics (5-Year)</h2>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                  <div>
                    <p className="text-sm text-gray-600">Current P/E</p>
                    <p className="text-lg font-semibold">{peStats.current?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Average P/E</p>
                    <p className="text-lg font-semibold">{peStats.average?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Median P/E</p>
                    <p className="text-lg font-semibold">{peStats.median?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Min P/E</p>
                    <p className="text-lg font-semibold">{peStats.min?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Max P/E</p>
                    <p className="text-lg font-semibold">{peStats.max?.toFixed(2) || "N/A"}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Charts Section */}
            <div className="space-y-6">
              {chartData && (
                <>
                  <div className="bg-white rounded-lg shadow-md p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xl font-semibold text-gray-900">{selectedSymbol} - Price Chart</h2>
                      <div>
                        <label htmlFor="priceChartType" className="mr-2 text-sm text-gray-700">Chart Type:</label>
                        <select
                          id="priceChartType"
                          value={priceChartType}
                          onChange={e => setPriceChartType(e.target.value as any)}
                          className="px-2 py-1 border border-gray-300 rounded"
                        >
                          <option value="candlestick">Candlestick</option>
                          <option value="line">Line</option>
                          <option value="area">Area</option>
                        </select>
                      </div>
                    </div>
                    <Plot
                      data={chartData.priceData}
                      layout={chartData.priceLayout}
                      config={{ responsive: true }}
                      style={{ width: "100%", height: "500px" }}
                    />
                  </div>
                  <div className="bg-white rounded-lg shadow-md p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xl font-semibold text-gray-900">{selectedSymbol} - Volume Chart</h2>
                      <div>
                        <label htmlFor="volumeChartType" className="mr-2 text-sm text-gray-700">Chart Type:</label>
                        <select
                          id="volumeChartType"
                          value={volumeChartType}
                          onChange={e => setVolumeChartType(e.target.value as any)}
                          className="px-2 py-1 border border-gray-300 rounded"
                        >
                          <option value="bar">Bar</option>
                          <option value="line">Line</option>
                          <option value="area">Area</option>
                        </select>
                      </div>
                    </div>
                    <Plot
                      data={chartData.volumeData}
                      layout={chartData.volumeLayout}
                      config={{ responsive: true }}
                      style={{ width: "100%", height: "300px" }}
                    />
                  </div>
                </>
              )}
              {peChartData && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-xl font-semibold text-gray-900 mb-4">{selectedSymbol} - Historical P/E Ratio</h2>
                  <Plot
                    data={peChartData.data}
                    layout={peChartData.layout}
                    config={{ responsive: true }}
                    style={{ width: "100%", height: "400px" }}
                  />
                </div>
              )}
            </div>
          </>
        )}

        {/* Popular Stocks Quick Access */}
        <div className="bg-white rounded-lg shadow-md p-6 mt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Popular Stocks</h3>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            {["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"].map((symbol) => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                disabled={isLoading}
                className={`p-3 rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${selectedSymbol === symbol ? "border-blue-500 bg-blue-50 text-blue-700" : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"}`}
              >
                <div className="font-semibold">{symbol}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Helper functions
function getPeriodStartDate(period: string): Date {
  const now = new Date()
  const startDate = new Date(now)
  switch (period) {
    case "1mo":
      startDate.setMonth(now.getMonth() - 1)
      break
    case "3mo":
      startDate.setMonth(now.getMonth() - 3)
      break
    case "6mo":
      startDate.setMonth(now.getMonth() - 6)
      break
    case "1y":
      startDate.setFullYear(now.getFullYear() - 1)
      break
    case "2y":
      startDate.setFullYear(now.getFullYear() - 2)
      break
    case "5y":
      startDate.setFullYear(now.getFullYear() - 5)
      break
    default:
      startDate.setFullYear(now.getFullYear() - 1)
  }
  return startDate
}

function formatMarketCap(marketCap: number): string {
  if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`
  if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`
  if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`
  return `$${marketCap.toLocaleString()}`
}

function getChartData(
  historicalData: HistoricalData[],
  symbol: string,
  priceChartType: 'candlestick' | 'line' | 'area' = 'candlestick',
  volumeChartType: 'bar' | 'line' | 'area' = 'bar'
) {
  if (!historicalData || historicalData.length === 0) return null

  const dates = historicalData.map((item) => item.date)
  const opens = historicalData.map((item) => item.open)
  const highs = historicalData.map((item) => item.high)
  const lows = historicalData.map((item) => item.low)
  const closes = historicalData.map((item) => item.close)
  const volumes = historicalData.map((item) => item.volume)

  // Price chart data
  let priceData: PlotData[] = []
  if (priceChartType === "candlestick") {
    priceData = [
      {
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        type: "candlestick",
        name: symbol,
        increasing: { line: { color: "#00C851" } },
        decreasing: { line: { color: "#FF4444" } },
      },
    ]
  } else if (priceChartType === "line") {
    priceData = [
      {
        x: dates,
        y: closes,
        type: "scatter",
        mode: "lines",
        name: symbol,
        line: { color: "#007bff" },
      },
    ]
  } else if (priceChartType === "area") {
    priceData = [
      {
        x: dates,
        y: closes,
        type: "scatter",
        mode: "lines",
        fill: "tozeroy",
        name: symbol,
        line: { color: "#007bff" },
      },
    ]
  }

  const priceLayout: Partial<Layout> = {
    title: `${symbol} Stock Price`,
    xaxis: { type: "date" },
    yaxis: { title: "Price ($)" },
    margin: { t: 50, r: 50, b: 50, l: 80 },
  }

  // Volume chart data
  let volumeData: PlotData[] = []
  if (volumeChartType === "bar") {
    volumeData = [
      { x: dates, y: volumes, type: "bar", name: "Volume", marker: { color: "#007bff" } },
    ]
  } else if (volumeChartType === "line") {
    volumeData = [
      { x: dates, y: volumes, type: "scatter", mode: "lines", name: "Volume", line: { color: "#00C851" } },
    ]
  } else if (volumeChartType === "area") {
    volumeData = [
      { x: dates, y: volumes, type: "scatter", mode: "lines", fill: "tozeroy", name: "Volume", line: { color: "#00C851" } },
    ]
  }

  const volumeLayout: Partial<Layout> = {
    title: `${symbol} Trading Volume`,
    xaxis: { type: "date" },
    yaxis: { title: "Volume" },
    margin: { t: 50, r: 50, b: 50, l: 80 },
  }

  return { priceData, priceLayout, volumeData, volumeLayout }
}

function getPeChartData(peData: PERatioPoint[], symbol: string) {
  if (!peData || peData.length === 0) return null

  const validPeData = peData.filter((d) => d.peRatio !== null)
  if (validPeData.length === 0) return null

  const dates = validPeData.map((item) => item.date)
  const peRatios = validPeData.map((item) => item.peRatio)

  const data: PlotData[] = [{ x: dates, y: peRatios, type: "scatter", mode: "lines+markers", name: "P/E Ratio" }]
  const layout: Partial<Layout> = {
    title: `${symbol} Historical P/E Ratio`,
    xaxis: { type: "date" },
    yaxis: { title: "P/E Ratio" },
    margin: { t: 50, r: 50, b: 50, l: 80 },
  }

  return { data, layout }
}
