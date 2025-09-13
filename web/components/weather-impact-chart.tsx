"use client"

import { Line, LineChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface WeatherImpactChartProps {
  facilityId: string
}

export default function WeatherImpactChart({ facilityId }: WeatherImpactChartProps) {
  // Generate sample data for weather impact
  const generateWeatherImpactData = () => {
    const data = []
    const now = new Date()
    const oneMonthAgo = new Date()
    oneMonthAgo.setMonth(now.getMonth() - 1)

    // Generate daily data points for 1 month (30 days)
    for (let i = 0; i < 30; i++) {
      const date = new Date(oneMonthAgo)
      date.setDate(oneMonthAgo.getDate() + i)

      // Temperature varies between 50-90°F with some randomness
      const temperature = Math.round(70 + Math.sin((i / 30) * Math.PI * 2) * 15 + (Math.random() * 10 - 5))

      // Emissions correlate somewhat with temperature
      const tempFactor = (temperature - 50) / 40 // Normalize to 0-1 range
      const baseEmissions = facilityId === "1" ? 2200 : 2000
      const emissions = Math.round(baseEmissions + tempFactor * 800 + (Math.random() * 400 - 200))

      data.push({
        date: date.toISOString().split("T")[0],
        emissions: emissions,
        temperature: temperature,
      })
    }

    return data
  }

  const weatherImpactData = generateWeatherImpactData()

  return (
    <ChartContainer
      config={{
        emissions: {
          label: "Methane Emissions",
          color: "hsl(var(--chart-1))",
        },
        temperature: {
          label: "Temperature",
          color: "hsl(var(--chart-2))",
        },
      }}
    >
      <LineChart
        accessibilityLayer
        data={weatherImpactData}
        margin={{
          left: 5,
          right: 5,
          top: 5,
          bottom: 5,
        }}
      >
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          tickFormatter={(value) => {
            const date = new Date(value)
            return `${date.getDate()}/${date.getMonth() + 1}`
          }}
          tick={{ fontSize: 12 }}
          minTickGap={30}
        />
        <YAxis
          yAxisId="left"
          orientation="left"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          domain={["auto", "auto"]}
          tick={{ fontSize: 12 }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          domain={[40, 100]}
          tick={{ fontSize: 12 }}
          unit="°F"
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              formatter={(value, name) => {
                if (name === "emissions") return `${value.toLocaleString()} BTU`
                if (name === "temperature") return `${value}°F`
                return value
              }}
              labelFormatter={(label) => {
                const date = new Date(label)
                return date.toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })
              }}
            />
          }
        />
        <Line
          type="monotone"
          dataKey="emissions"
          stroke="var(--color-emissions)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
          yAxisId="left"
        />
        <Line
          type="monotone"
          dataKey="temperature"
          stroke="var(--color-temperature)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
          yAxisId="right"
        />
      </LineChart>
    </ChartContainer>
  )
}
