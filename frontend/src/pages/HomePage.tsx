import React from 'react';
import { Grid, Typography } from '@mui/material';
import WeatherCard from '../components/WeatherCard';
import AttractionCard from '../components/AttractionCard';
import attractionsData from '../data/attractions.json';
import { Attraction } from '../types';
import { useWeather } from '../hooks/useWeather';

const HomePage: React.FC = () => {
  const attractions = attractionsData.attractions as Attraction[];
  const { weather, loading, error } = useWeather();
  
  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Welcome to wAItless
      </Typography>
      <Typography variant="subtitle1" paragraph>
        Your real-time guide to Europa-Park attractions and wait times.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <WeatherCard weather={weather} loading={loading} error={error} />
        </Grid>

        {attractions.map((attraction) => (
          <Grid item xs={12} md={4} key={attraction.id}>
            <AttractionCard attraction={attraction} />
          </Grid>
        ))}
      </Grid>
    </div>
  );
};

export default HomePage; 