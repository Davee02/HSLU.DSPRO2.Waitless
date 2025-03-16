import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress, Divider, Grid } from '@mui/material';
import { WbSunny, Cloud, Grain, Thunderstorm, AcUnit, WaterDrop } from '@mui/icons-material';
import { Weather } from '../types';

interface WeatherCardProps {
  weather: Weather | null;
  loading?: boolean;
  error?: string | null;
}

const WeatherCard: React.FC<WeatherCardProps> = ({ weather, loading = false, error = null }) => {
  const getWeatherIcon = (condition: string, size: number = 40) => {
    switch (condition.toLowerCase()) {
      case 'clear':
      case 'sunny':
        return <WbSunny sx={{ fontSize: size }} />;
      case 'partly cloudy':
      case 'cloudy':
        return <Cloud sx={{ fontSize: size }} />;
      case 'rain':
      case 'rain showers':
      case 'drizzle':
        return <WaterDrop sx={{ fontSize: size }} />;
      case 'snow':
        return <AcUnit sx={{ fontSize: size }} />;
      case 'thunderstorm':
        return <Thunderstorm sx={{ fontSize: size }} />;
      default:
        return <Grain sx={{ fontSize: size }} />;
    }
  };

  const formatHour = (hour: number) => {
    return hour.toString().padStart(2, '0') + ':00';
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography variant="h6" gutterBottom>
          Current Weather in Rust
        </Typography>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Typography color="error">{error}</Typography>
        ) : weather ? (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
              {getWeatherIcon(weather.current.condition)}
              <Box>
                <Typography variant="h4">{weather.current.temperature}°C</Typography>
                <Typography variant="body1" color="text.secondary">
                  {weather.current.condition}
                </Typography>
              </Box>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle1" gutterBottom>
              6-Hour Forecast
            </Typography>
            <Grid container spacing={1}>
              {weather.forecast.map((item) => (
                <Grid item xs={2} key={item.time}>
                  <Box sx={{ textAlign: 'center', p: 1 }}>
                    <Typography variant="caption" display="block" sx={{ mb: 0.5 }}>
                      {formatHour(item.time)}
                    </Typography>
                    {getWeatherIcon(item.condition, 24)}
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                      {item.temperature}°C
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </>
        ) : (
          <Typography>No weather data available</Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default WeatherCard; 