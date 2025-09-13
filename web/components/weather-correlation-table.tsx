"use client"

export default function WeatherCorrelationTable() {
  const weatherConditions = [
    { code: 0, symbol: "☀️", name: "Clear Sky", value: 1.0 },
    { code: 1, symbol: "🌤️", name: "Mainly Clear", value: 0.95 },
    { code: 2, symbol: "⛅", name: "Partly Cloudy", value: 0.9 },
    { code: 3, symbol: "☁️", name: "Overcast", value: 0.85 },
    { code: 45, symbol: "🌫️", name: "Fog", value: 0.7 },
    { code: 51, symbol: "🌧️", name: "Light Drizzle", value: 0.8 },
    { code: 53, symbol: "🌧️", name: "Moderate Drizzle", value: 0.75 },
    { code: 61, symbol: "🌧️", name: "Light Rain", value: 0.75 },
    { code: 63, symbol: "🌧️", name: "Moderate Rain", value: 0.7 },
    { code: 65, symbol: "🌧️", name: "Heavy Rain", value: 0.6 },
    { code: 71, symbol: "❄️", name: "Light Snow", value: 0.65 },
    { code: 73, symbol: "❄️", name: "Moderate Snow", value: 0.6 },
    { code: 95, symbol: "⛈️", name: "Thunderstorm", value: 0.5 },
  ]

  return (
    <div className="bg-muted/50 p-4 rounded h-80 overflow-y-auto">
      <table className="w-full">
        <thead className="bg-muted/50">
          <tr>
            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Weather</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Avg. Energy</th>
            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Impact</th>
          </tr>
        </thead>
        <tbody>
          {weatherConditions.map((weather, index) => (
            <tr key={weather.code} className={index % 2 === 0 ? "" : "bg-muted/30"}>
              <td className="px-3 py-2 whitespace-nowrap">
                <div className="flex items-center">
                  <span className="mr-2">{weather.symbol}</span>
                  <span className="text-sm">{weather.name}</span>
                </div>
              </td>
              <td className="px-3 py-2 whitespace-nowrap text-sm">{(10000 * weather.value).toLocaleString()} BTU</td>
              <td className="px-3 py-2 whitespace-nowrap">
                <div className="flex items-center">
                  <div className="w-16 bg-muted rounded-full h-2">
                    <div className="bg-green-600 h-2 rounded-full" style={{ width: `${weather.value * 100}%` }}></div>
                  </div>
                  <span className="ml-2 text-sm">{Math.round(weather.value * 100)}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
