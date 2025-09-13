// Sample data generation functions

export async function fetchWeeklyPredictions() {
  // In a real application, this would be an API call
  // For now, we'll generate sample data

  const generateWeeklyData = () => {
    const data = []
    const now = new Date()
    const twoYearsAgo = new Date()
    twoYearsAgo.setFullYear(now.getFullYear() - 2)

    // Generate weekly data points for 2 years (104 weeks)
    for (let i = 0; i < 104; i++) {
      const date = new Date(twoYearsAgo)
      date.setDate(date.getDate() + i * 7)

      // Add some randomness and seasonal patterns
      const seasonalFactor = Math.sin((i / 52) * Math.PI) * 0.3 + 1

      data.push({
        date: date.toISOString().split("T")[0],
        facility1: Math.round((Math.random() * 500 + 99000) * seasonalFactor),
        facility2: Math.round((Math.random() * 400 + 88000) * seasonalFactor),
        temperature: Math.round(70 + Math.sin((i / 52) * Math.PI * 2) * 15 + (Math.random() * 10 - 5)),
        humidity: Math.round(65 + Math.sin((i / 52) * Math.PI * 2) * 10 + (Math.random() * 8 - 4)),
        methaneLevel: Math.round(65 + Math.sin((i / 52) * Math.PI) * 5 + (Math.random() * 6 - 3)),
      })
    }

    return data
  }

  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  return generateWeeklyData()
}

export async function fetchMonthlyPredictions() {
  // In a real application, this would be an API call
  // For now, we'll generate sample data

  const generateMonthlyData = () => {
    const data = []
    const now = new Date()
    const twoYearsAgo = new Date()
    twoYearsAgo.setFullYear(now.getFullYear() - 2)

    // Generate monthly data points for 2 years (24 months)
    for (let i = 0; i < 24; i++) {
      const date = new Date(twoYearsAgo)
      date.setMonth(twoYearsAgo.getMonth() + i)

      // Add some randomness and seasonal patterns
      const seasonalFactor = Math.sin((i / 12) * Math.PI * 2) * 0.3 + 1

      data.push({
        month: date.toISOString().split("T")[0].substring(0, 7),
        facility1: Math.round((Math.random() * 60000 + 80000) * seasonalFactor),
        facility2: Math.round((Math.random() * 18000 + 70000) * seasonalFactor),
      })
    }

    return data
  }

  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  return generateMonthlyData()
}

export async function fetchPredictionForNextDays(days = 90) {
  // Generate prediction data for the next X days
  const data = []
  const now = new Date()

  for (let i = 0; i < days; i++) {
    const date = new Date(now)
    date.setDate(date.getDate() + i)

    // Add some randomness and seasonal patterns
    const seasonalFactor = Math.sin((i / 90) * Math.PI) * 0.2 + 1

    data.push({
      date: date.toISOString().split("T")[0],
      facility1: Math.round((Math.random() * 500 + 121000) * seasonalFactor),
      facility2: Math.round((Math.random() * 400 + 129000) * seasonalFactor),
      temperature: Math.round(70 + Math.sin((i / 90) * Math.PI * 2) * 15 + (Math.random() * 10 - 5)),
      humidity: Math.round(65 + Math.sin((i / 90) * Math.PI * 2) * 10 + (Math.random() * 8 - 4)),
    })
  }

  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 800))

  return data
}
