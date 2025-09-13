"use client"

export default function FeatureImportanceChart() {
  const featureImportance = [
    { name: "Temperature", importance: 0.32 },
    { name: "Flow Rate", importance: 0.28 },
    { name: "Methane Concentration", importance: 0.18 },
    { name: "Humidity", importance: 0.12 },
    { name: "Pressure", importance: 0.06 },
    { name: "Wind Speed", importance: 0.04 },
  ]

  return (
    <div className="space-y-4">
      {featureImportance.map((feature) => (
        <div key={feature.name}>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm font-medium">{feature.name}</span>
            <span className="text-sm text-muted-foreground">{feature.importance.toFixed(3)}</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div className="bg-green-600 h-2 rounded-full" style={{ width: `${feature.importance * 100}%` }}></div>
          </div>
        </div>
      ))}
    </div>
  )
}
