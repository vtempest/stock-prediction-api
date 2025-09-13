import { NextResponse } from "next/server"
import yahooFinance from "yahoo-finance2"
import { HistoricalPECalculator } from "@/lib/historical-pe-calculator"

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

export async function GET(request: Request, { params }: { params: { symbol: string } }) {
  const { symbol } = params
  const { searchParams } = new URL(request.url)
  const period = searchParams.get("period") || "1y"

  if (!symbol) {
    return NextResponse.json({ error: "Symbol parameter is required" }, { status: 400 })
  }

  try {
    const startDate = getPeriodStartDate(period)
    const endDate = new Date()

    const peCalculator = new HistoricalPECalculator()

    // Use Promise.allSettled to ensure we get as much data as possible
    const results = await Promise.allSettled([
      yahooFinance.quote(symbol),
      yahooFinance.chart(symbol, {
        period1: startDate.toISOString().split("T")[0],
        period2: endDate.toISOString().split("T")[0],
        interval: "1d",
      }),
      peCalculator.calculateHistoricalPEForStock(
        symbol,
        // For P/E, always use a longer history for better calculation
        new Date(new Date().setFullYear(new Date().getFullYear() - 5)).toISOString().split("T")[0],
        endDate.toISOString().split("T")[0],
        "1mo",
      ),
    ])

    const quoteResult = results[0].status === "fulfilled" ? results[0].value : null
    const historicalResult = results[1].status === "fulfilled" ? results[1].value : null
    const peResult = results[2].status === "fulfilled" ? results[2].value : null

    const errors = results
      .map((r, i) =>
        r.status === "rejected" ? { source: ["quote", "historical", "pe"][i], reason: r.reason.message } : null,
      )
      .filter(Boolean)

    if (!quoteResult && !historicalResult && !peResult) {
      return NextResponse.json({ error: "Failed to fetch any data for the symbol.", details: errors }, { status: 500 })
    }

    return NextResponse.json({
      quote: quoteResult,
      historical: historicalResult?.quotes || [],
      pe: peResult,
      errors,
    })
  } catch (error) {
    console.error(`Critical error in stock-analysis for ${symbol}:`, error)
    const errorMessage = error instanceof Error ? error.message : "An unknown critical error occurred"
    return NextResponse.json({ error: "Failed to process request", details: errorMessage }, { status: 500 })
  }
}
