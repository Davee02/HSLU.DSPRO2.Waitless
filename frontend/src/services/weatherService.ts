const RUST_LATITUDE = 48.266;
const RUST_LONGITUDE = 7.722;

const getConditionFromCode = (weatherCode: number): string => {
  if ([0].includes(weatherCode)) return 'Clear';
  if ([1, 2, 3].includes(weatherCode)) return 'Partly Cloudy';
  if ([45, 48].includes(weatherCode)) return 'Foggy';
  if ([51, 53, 55, 56, 57].includes(weatherCode)) return 'Drizzle';
  if ([61, 63, 65, 66, 67].includes(weatherCode)) return 'Rain';
  if ([71, 73, 75, 77].includes(weatherCode)) return 'Snow';
  if ([80, 81, 82].includes(weatherCode)) return 'Rain Showers';
  if ([95].includes(weatherCode)) return 'Thunderstorm';
  return 'Unknown';
};

export const fetchWeather = async () => {
  try {
    const response = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${RUST_LATITUDE}&longitude=${RUST_LONGITUDE}&current=temperature_2m,weather_code&hourly=temperature_2m,weather_code&timezone=Europe/Berlin`
    );
    const data = await response.json();

    // For debugging
    console.log('Raw weather data:', data);

    // Current weather
    const currentCondition = getConditionFromCode(data.current.weather_code);
    const currentHour = new Date().getHours();

    // Find the starting index for the current hour
    const currentTimeIndex = data.hourly.time.findIndex((time: string) => {
      const hour = new Date(time).getHours();
      return hour >= currentHour;
    });

    // Hourly forecast starting from the next hour
    const hourlyForecast = data.hourly.time
      .slice(currentTimeIndex + 1, currentTimeIndex + 7)
      .map((time: string, index: number) => {
        const actualIndex = currentTimeIndex + 1 + index;
        return {
          time: new Date(time).getHours(),
          temperature: Math.round(data.hourly.temperature_2m[actualIndex]),
          condition: getConditionFromCode(data.hourly.weather_code[actualIndex]),
        };
      });

    // For debugging
    console.log('Processed forecast:', hourlyForecast);

    return {
      current: {
        temperature: Math.round(data.current.temperature_2m),
        condition: currentCondition,
        icon: currentCondition.toLowerCase().replace(' ', '-')
      },
      forecast: hourlyForecast
    };
  } catch (error) {
    console.error('Error fetching weather:', error);
    throw error;
  }
}; 