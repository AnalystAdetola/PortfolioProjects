-- Cleansed DIM_Calender -- 
SELECT 
  TOP (1000) [DateKey], 
  [FullDateAlternateKey] AS Date, 
  --,[DayNumberOfWeek]
  [EnglishDayNameOfWeek] AS Day,
  --,[SpanishDayNameOfWeek]
  --,[FrenchDayNameOfWeek]
  --,[DayNumberOfMonth]
  --,[DayNumberOfYear]
  [WeekNumberOfYear] AS WeekNº, 
  [EnglishMonthName] AS Month,
  LEFT([EnglishMonthName], 3) AS MonthShort,
  --,[SpanishMonthName]
  --,[FrenchMonthName]
  [MonthNumberOfYear] AS MonthNº, 
  [CalendarQuarter] AS Quarter,
  [CalendarYear] AS Year
  --,[CalendarSemester]
  --,[FiscalQuarter]
  --,[FiscalYear]
  --,[FiscalSemester]
FROM 
  [AdventureWorksDW2022].[dbo].[DimDate]
WHERE 
  CalendarYear >= 2023 