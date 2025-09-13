"use client"

interface PredictionResult {
  date: string
  facility1: number
  facility2: number
  total: number
}

interface PredictionResultsTableProps {
  data: PredictionResult[]
}

export default function PredictionResultsTable({ data }: PredictionResultsTableProps) {
  return (
    <table className="w-full">
      <thead className="bg-muted/50">
        <tr>
          <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Date</th>
          <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">Facility 1</th>
          <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">Facility 2</th>
          <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">Total</th>
        </tr>
      </thead>
      <tbody>
        {data.map((day, i) => (
          <tr key={day.date} className={i % 2 === 0 ? "" : "bg-muted/30"}>
            <td className="px-3 py-2 whitespace-nowrap text-sm">{new Date(day.date).toLocaleDateString()}</td>
            <td className="px-3 py-2 whitespace-nowrap text-sm text-right">{day.facility1.toLocaleString()} BTU</td>
            <td className="px-3 py-2 whitespace-nowrap text-sm text-right">{day.facility2.toLocaleString()} BTU</td>
            <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-right">
              {day.total.toLocaleString()} BTU
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
