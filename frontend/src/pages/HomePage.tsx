import React, { useMemo } from 'react';
import { Grid, Typography, Container, Box } from '@mui/material';
import WeatherCard from '../components/WeatherCard';
import AttractionCard from '../components/AttractionCard';
import { Attraction } from '../types';
import { useWeather } from '../hooks/useWeather';

// Import attractions data
import attractionsJson from '../data/attractions.json';

// Import all attraction images
const getImageUrl = (imageName: string) => {
  try {
    return require(`../img/attractions/${imageName}`);
  } catch {
    return undefined;
  }
};

const HomePage: React.FC = () => {
  const { weather, loading, error } = useWeather();

  // Get 5 featured attractions (mix of thrill and family rides)
  const featuredAttractions = useMemo(() => {
    const attractions = attractionsJson.attractions as unknown as Attraction[];
    const thrillRides = attractions.filter(a => a.category === 'thrill').slice(0, 3);
    const familyRides = attractions.filter(a => a.category === 'family').slice(0, 2);
    
    // Fix image paths
    return [...thrillRides, ...familyRides].map(attraction => ({
      ...attraction,
      imageUrl: getImageUrl(attraction.imageUrl.split('/').pop() || '')
    }));
  }, []);
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Welcome to wAItless
        </Typography>
        <Typography variant="h5" align="center" color="text.secondary" paragraph sx={{ mb: 6 }}>
          Your real-time guide to Europa-Park attractions and wait times
        </Typography>

        <Grid container spacing={3}>
          {/* First row */}
          <Grid item xs={12} sm={6} md={4}>
            <WeatherCard weather={weather} loading={loading} error={error} />
          </Grid>
          {featuredAttractions.slice(0, 2).map((attraction) => (
            <Grid item xs={12} sm={6} md={4} key={attraction.id}>
              <AttractionCard 
                attraction={{
                  ...attraction,
                  description: attraction.short_description
                }} 
              />
            </Grid>
          ))}
          
          {/* Second row */}
          {featuredAttractions.slice(2).map((attraction) => (
            <Grid item xs={12} sm={6} md={4} key={attraction.id}>
              <AttractionCard 
                attraction={{
                  ...attraction,
                  description: attraction.short_description
                }} 
              />
            </Grid>
          ))}
        </Grid>
      </Box>
    </Container>
  );
};

export default HomePage; 