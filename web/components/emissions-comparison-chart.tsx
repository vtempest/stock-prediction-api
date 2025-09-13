"use client"

import { Line } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { CartesianGrid, XAxis, YAxis, LineChart } from "recharts"

export default function StockPriceComparisonChart() {
  // Generate sample data for the last 30 days for two stocks
  const generateStockData = () => {
    const data = []
    const now = new Date()
    const thirtyDaysAgo = new Date()
    thirtyDaysAgo.setDate(now.getDate() - 30)
    const base1 = 180 + Math.random() * 20
    const base2 = 320 + Math.random() * 30
    for (let i = 0; i < 30; i++) {
      const date = new Date(thirtyDaysAgo)
      date.setDate(date.getDate() + i)
      // Simulate price movement
      const price1 = base1 + Math.sin(i / 5) * 5 + Math.random() * 2
      const price2 = base2 + Math.cos(i / 6) * 7 + Math.random() * 3
      data.push({
        date: date.toISOString().split("T")[0],
        AAPL: parseFloat(price1.toFixed(2)),
        MSFT: parseFloat(price2.toFixed(2)),
      })
    }
    return data
  }
  const stockData = generateStockData()
  return (
    <ChartContainer
      config={{
        AAPL: {
          label: "AAPL",
          color: "hsl(var(--chart-1))",
        },
        MSFT: {
          label: "MSFT",
          color: "hsl(var(--chart-2))",
        },
      }}
    >
      <LineChart
        accessibilityLayer
        data={stockData}
        margin={{ left: 5, right: 5, top: 5, bottom: 5 }}
      >
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          tickFormatter={(value) => {
            const date = new Date(value)
            return `${date.toLocaleString("default", { month: "short" })} ${date.getFullYear()}`
          }}
          tick={{ fontSize: 12 }}
          minTickGap={30}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          tickFormatter={(value) => `$${value.toFixed(2)}`}
          tick={{ fontSize: 12 }}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              formatter={(value) => `$${value.toLocaleString()}`}
              labelFormatter={(label) => {
                const date = new Date(label)
                return date.toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })
              }}
            />
          }
        />
        <Line
          type="monotone"
          dataKey="AAPL"
          stroke="var(--color-stock1)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
        />
        <Line
          type="monotone"
          dataKey="MSFT"
          stroke="var(--color-stock2)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ChartContainer>
  )
}
