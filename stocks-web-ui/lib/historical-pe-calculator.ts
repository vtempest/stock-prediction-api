import yahooFinance from "yahoo-finance2"

// Define interfaces for our data structures
interface PricePoint {
  date: Date
  close: number
  adjClose: number
}

interface EarningsPeriod {
  date: Date
  eps: number | null
  netIncome: number | null
  sharesOutstanding: number | null
  period: "quarterly" | "annual" | "ttm_current"
}

interface PERatioPoint {
  date: Date
  price: number
  ttmEPS: number | null
  peRatio: number | null
}

export class HistoricalPECalculator {
  private priceData: PricePoint[] = []
  private earningsData: EarningsPeriod[] = []
  public peRatios: PERatioPoint[] = []

  /**
   * Fetch historical price data for a given symbol
   */
  async fetchHistoricalPrices(
    symbol: string,
    startDate: string,
    endDate: string,
    interval: "1d" | "1wk" | "1mo" = "1d",
  ) {
    try {
      console.log(`Fetching historical prices for ${symbol}...`)
      const queryOptions = { period1: startDate, period2: endDate, interval }
      const result = await yahooFinance.chart(symbol, queryOptions)
      const quotes = result.quotes || []
      this.priceData = quotes.map((item) => ({
        date: new Date(item.date),
        close: item.close,
        adjClose: item.adjclose || item.close,
      }))
      console.log(`Fetched ${this.priceData.length} price data points`)
      if (this.priceData.length > 0) {
        console.log(
          `Price data range: ${this.priceData[0]?.date.toISOString().split("T")[0]} to ${this.priceData[this.priceData.length - 1]?.date.toISOString().split("T")[0]}`,
        )
      }
      return this.priceData
    } catch (error) {
      console.error("Error fetching historical prices:", error)
      throw error
    }
  }

  /**
   * Fetch earnings data for calculating TTM EPS
   */
  async fetchEarningsData(symbol: string) {
    try {
      console.log(`Fetching earnings data for ${symbol}...`)
      let earningsData: any = null

      // Attempt to fetch quarterly earnings
      try {
        const quarterlyResult = await yahooFinance.quoteSummary(symbol, {
          modules: ["incomeStatementHistoryQuarterly"],
        })
        if (quarterlyResult?.incomeStatementHistoryQuarterly?.incomeStatementHistory?.length) {
          earningsData = {
            quarterly: quarterlyResult.incomeStatementHistoryQuarterly.incomeStatementHistory,
            type: "quarterly",
          }
        }
      } catch (error) {
        console.log("Quarterly earnings fetch error:", (error as Error).message)
      }

      // Fallback to annual if quarterly fails
      if (!earningsData) {
        try {
          const annualResult = await yahooFinance.quoteSummary(symbol, { modules: ["incomeStatementHistory"] })
          if (annualResult?.incomeStatementHistory?.incomeStatementHistory?.length) {
            earningsData = { annual: annualResult.incomeStatementHistory.incomeStatementHistory, type: "annual" }
          }
        } catch (error) {
          console.log("Annual earnings fetch error:", (error as Error).message)
        }
      }

      // Fallback to earnings module
      if (!earningsData) {
        try {
          const earningsResult = await yahooFinance.quoteSummary(symbol, { modules: ["earnings"] })
          if (earningsResult?.earnings?.earningsChart?.quarterly?.length) {
            earningsData = { earnings: earningsResult.earnings, type: "earnings" }
          }
        } catch (error) {
          console.log("Earnings module fetch error:", (error as Error).message)
        }
      }

      if (!earningsData) {
        throw new Error("Could not fetch earnings data from any module")
      }

      this.earningsData = this.processEarningsData(earningsData)
      console.log(`Processed ${this.earningsData.length} earnings periods`)
      return this.earningsData
    } catch (error) {
      console.error("Error fetching earnings data:", error)
      throw error
    }
  }

  /**
   * Fallback method: Use current TTM EPS for all historical calculations
   */
  async fetchCurrentEPSFallback(symbol: string): Promise<boolean> {
    try {
      console.log("\n=== Using Fallback Method: Current TTM EPS ===")
      const currentData = await yahooFinance.quoteSummary(symbol, {
        modules: ["defaultKeyStatistics", "financialData"],
      })
      let currentTTMEPS: number | null = null

      if (currentData.defaultKeyStatistics?.trailingEps?.raw) {
        currentTTMEPS = currentData.defaultKeyStatistics.trailingEps.raw
      } else if (currentData.financialData?.earningsPerShare?.raw) {
        currentTTMEPS = currentData.financialData.earningsPerShare.raw
      }

      if (currentTTMEPS && currentTTMEPS > 0) {
        this.earningsData = [
          {
            date: new Date("2000-01-01"), // Use old date to apply to all price data
            eps: currentTTMEPS,
            netIncome: null,
            sharesOutstanding: null,
            period: "ttm_current",
          },
        ]
        console.log(`Created fallback earnings data with TTM EPS: ${currentTTMEPS}`)
        return true
      }
      return false
    } catch (error) {
      console.log("Fallback method also failed:", (error as Error).message)
      return false
    }
  }

  /**
   * Process raw earnings data into a standardized format
   */
  processEarningsData(rawData: any): EarningsPeriod[] {
    let processedData: EarningsPeriod[] = []
    if (rawData.type === "quarterly" && rawData.quarterly) {
      processedData = rawData.quarterly
        .map((s: any) => ({
          date: s.endDate?.raw ? new Date(s.endDate.raw * 1000) : null,
          eps: s.basicEPS?.raw || s.dilutedEPS?.raw || null,
          netIncome: s.netIncome?.raw || null,
          sharesOutstanding: s.weightedAverageShsOut?.raw || s.weightedAverageShsOutDil?.raw || null,
          period: "quarterly",
        }))
        .filter((item: any) => item.date && item.eps !== null)
    } else if (rawData.type === "annual" && rawData.annual) {
      processedData = rawData.annual
        .map((s: any) => ({
          date: s.endDate?.raw ? new Date(s.endDate.raw * 1000) : null,
          eps: s.basicEPS?.raw || s.dilutedEPS?.raw || null,
          netIncome: s.netIncome?.raw || null,
          sharesOutstanding: s.weightedAverageShsOut?.raw || s.weightedAverageShsOutDil?.raw || null,
          period: "annual",
        }))
        .filter((item: any) => item.date && item.eps !== null)
    } else if (rawData.type === "earnings" && rawData.earnings) {
      if (rawData.earnings.earningsChart?.quarterly) {
        processedData = rawData.earnings.earningsChart.quarterly
          .map((q: any) => ({
            date: q.date ? new Date(q.date) : null,
            eps: q.actual?.raw || q.estimate?.raw || null,
            netIncome: null,
            sharesOutstanding: null,
            period: "quarterly",
          }))
          .filter((item: any) => item.date && item.eps !== null)
      }
    }
    return processedData.sort((a, b) => a.date.getTime() - b.date.getTime())
  }

  /**
   * Calculate TTM EPS for a given date
   */
  calculateTTMEPS(targetDate: Date): number | null {
    const relevantEarnings = this.earningsData
      .filter((earning) => earning.date <= targetDate && earning.eps !== null)
      .sort((a, b) => b.date.getTime() - a.date.getTime())

    if (relevantEarnings.length === 0) return null

    if (relevantEarnings[0].period === "quarterly") {
      if (relevantEarnings.length < 4) return null // Need 4 quarters
      return relevantEarnings.slice(0, 4).reduce((sum, earning) => sum + (earning.eps || 0), 0)
    } else if (relevantEarnings[0].period === "annual") {
      return relevantEarnings[0].eps
    } else if (relevantEarnings[0].period === "ttm_current") {
      return relevantEarnings[0].eps
    }

    return null
  }

  /**
   * Calculate historical PE ratios
   */
  calculateHistoricalPE(): PERatioPoint[] {
    if (this.earningsData.length === 0) {
      console.log("No earnings data available for PE calculation")
      return []
    }
    this.peRatios = this.priceData.map((pricePoint) => {
      const ttmEPS = this.calculateTTMEPS(pricePoint.date)
      let peRatio = null
      if (ttmEPS && ttmEPS > 0) {
        peRatio = pricePoint.adjClose / ttmEPS
      }
      return {
        date: pricePoint.date,
        price: pricePoint.adjClose,
        ttmEPS: ttmEPS,
        peRatio: peRatio,
      }
    })
    return this.peRatios
  }

  /**
   * Get statistics about the historical PE ratios
   */
  getPEStatistics() {
    const validPEs = this.peRatios.filter((item) => item.peRatio !== null).map((item) => item.peRatio as number)
    if (validPEs.length === 0) return null
    const sorted = [...validPEs].sort((a, b) => a - b)
    return {
      count: validPEs.length,
      min: Math.min(...validPEs),
      max: Math.max(...validPEs),
      average: validPEs.reduce((sum, pe) => sum + pe, 0) / validPEs.length,
      median: sorted[Math.floor(sorted.length / 2)],
      current: this.peRatios[this.peRatios.length - 1]?.peRatio || null,
    }
  }

  /**
   * Main method to calculate historical PE ratios for a stock
   */
  async calculateHistoricalPEForStock(
    symbol: string,
    startDate: string,
    endDate: string,
    interval: "1d" | "1wk" | "1mo" = "1mo",
  ) {
    try {
      await this.fetchHistoricalPrices(symbol, startDate, endDate, interval)
      try {
        await this.fetchEarningsData(symbol)
      } catch (e) {
        console.log("Could not fetch detailed earnings, trying fallback...")
      }

      if (this.earningsData.length === 0) {
        const fallbackSuccess = await this.fetchCurrentEPSFallback(symbol)
        if (!fallbackSuccess) {
          throw new Error("Could not obtain any EPS data for PE calculation")
        }
      }

      this.calculateHistoricalPE()
      const statistics = this.getPEStatistics()

      return {
        priceData: this.priceData,
        earningsData: this.earningsData,
        peRatios: this.peRatios,
        statistics: statistics,
      }
    } catch (error) {
      console.error("Error in calculateHistoricalPEForStock:", error)
      throw error
    }
  }
}
