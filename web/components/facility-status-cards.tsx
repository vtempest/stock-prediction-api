"use client"

import { Card, CardContent } from "@/components/ui/card"
import { ArrowDown, ArrowUp, TrendingUp, TrendingDown } from "lucide-react"

export default function StockStatusCards() {
  const stocks = [
    {
      id: 1,
      symbol: "AAPL",
      name: "Apple Inc.",
      currentPrice: 192.34,
      change: 1.25,
      percentChange: 0.65,
      volume: 74200000,
      rsi: 54.2,
    },
    {
      id: 2,
      symbol: "MSFT",
      name: "Microsoft Corp.",
      currentPrice: 328.12,
      change: -2.18,
      percentChange: -0.66,
      volume: 31200000,
      rsi: 48.7,
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {stocks.map((stock) => (
        <Card
          key={stock.id}
          className={stock.id === 1 ? "border-blue-200 bg-blue-50" : "border-green-200 bg-green-50"}
        >
          <CardContent className="p-4">
            <div className="flex justify-between items-start">
              <h3 className={`font-medium text-lg ${stock.id === 1 ? "text-blue-800" : "text-green-800"}`}>
                {stock.symbol} <span className="text-xs text-muted-foreground ml-2">{stock.name}</span>
              </h3>
              <div
                className={`flex items-center text-sm font-medium ${stock.change >= 0 ? "text-green-600" : "text-red-600"}`}
              >
                {stock.change >= 0 ? <ArrowUp className="h-4 w-4 mr-1" /> : <ArrowDown className="h-4 w-4 mr-1" />}
                {stock.change >= 0 ? '+' : ''}{stock.change} ({stock.percentChange}%)
              </div>
            </div>

            <div className="mt-2">
              <div className="text-2xl font-bold">${stock.currentPrice.toLocaleString()}</div>
              <div className="text-sm text-muted-foreground">Current Price</div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="flex items-center">
                <TrendingUp className="h-4 w-4 mr-2 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">{stock.volume.toLocaleString()}</div>
                  <div className="text-xs text-muted-foreground">Volume</div>
                </div>
              </div>

              <div className="flex items-center">
                <TrendingDown className="h-4 w-4 mr-2 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">{stock.rsi}</div>
                  <div className="text-xs text-muted-foreground">RSI</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
