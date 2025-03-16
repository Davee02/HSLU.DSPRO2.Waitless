import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Chip,
  IconButton,
  Card,
  CardContent,
  Fade,
  Slide,
  Zoom,
  Divider,
  Button,
  Stack,
  Rating,
} from '@mui/material';
import {
  ArrowBack,
  Speed,
  Height,
  Timer,
  Warning,
  Category,
  CalendarToday,
  LocationOn,
  Person,
  Star,
  Info,
  AccessTime,
  DirectionsRun,
  Engineering,
} from '@mui/icons-material';

// Import attraction data
import attractionsData from '../data/attractions.json';

const AttractionPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [attraction, setAttraction] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Find the attraction by ID
    const found = attractionsData.attractions.find(a => a.id === id);
    if (found) {
      setAttraction(found);
    }
    setLoading(false);
  }, [id]);

  if (loading) {
    return <Box sx={{ p: 4, textAlign: 'center' }}>Loading...</Box>;
  }

  if (!attraction) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h5">Attraction not found</Typography>
        <Button onClick={() => navigate(-1)} startIcon={<ArrowBack />} sx={{ mt: 2, color: '#000000' }}>
          Go Back
        </Button>
      </Box>
    );
  }

  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'thrill':
        return '#f44336';
      case 'family':
        return '#4caf50';
      case 'children':
        return '#2196f3';
      case 'water':
        return '#00bcd4';
      case 'interactive':
        return '#ff9800';
      default:
        return '#9e9e9e';
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header Section */}
      <Box sx={{ mb: 4 }}>
        <Button
          onClick={() => navigate(-1)}
          startIcon={<ArrowBack sx={{ color: '#000000' }} />}
          sx={{ mb: 2, color: '#000000' }}
          variant="outlined"
        >
          Back to Park Information
        </Button>
        <Fade in timeout={1000}>
          <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
            {attraction.name}
          </Typography>
        </Fade>
        <Slide direction="right" in timeout={1000}>
          <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
            <Chip
              label={attraction.category.toUpperCase()}
              sx={{
                bgcolor: getCategoryColor(attraction.category),
                color: 'white',
                fontWeight: 'bold',
              }}
              icon={<Category sx={{ color: 'white' }} />}
            />
            <Chip
              label={attraction.area}
              variant="outlined"
              icon={<LocationOn />}
            />
          </Stack>
        </Slide>
      </Box>

      {/* Main Content Grid */}
      <Grid container spacing={4}>
        {/* Left Column - Main Info */}
        <Grid item xs={12} md={8}>
          <Fade in timeout={1000}>
            <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Info /> Description
              </Typography>
              <Typography variant="body1" paragraph>
                {attraction.description}
              </Typography>
            </Paper>
          </Fade>

          {/* Key Facts Grid */}
          <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>Key Facts</Typography>
          <Grid container spacing={2}>
            {attraction.keyFacts && Object.entries(attraction.keyFacts).map(([key, value]: [string, any]) => (
              <Grid item xs={12} sm={6} md={4} key={key}>
                <Zoom in timeout={1000}>
                  <Card elevation={3} sx={{
                    height: '100%',
                    transition: 'transform 0.2s',
                    '&:hover': {
                      transform: 'scale(1.02)',
                    },
                  }}>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                      </Typography>
                      <Typography variant="h6">
                        {value}
                      </Typography>
                    </CardContent>
                  </Card>
                </Zoom>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Right Column - Stats and Requirements */}
        <Grid item xs={12} md={4}>
          {/* Wait Time Card */}
          <Slide direction="left" in timeout={1000}>
            <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AccessTime /> Current Wait Time
              </Typography>
              <Typography variant="h3" sx={{ color: 'primary.main' }}>
                {attraction.waitTime || 'N/A'}
              </Typography>
              {attraction.predictedWaitTime && (
                <Typography variant="body2" color="text.secondary">
                  Predicted: {attraction.predictedWaitTime} minutes
                </Typography>
              )}
            </Paper>
          </Slide>

          {/* Requirements Card */}
          <Slide direction="left" in timeout={1000}>
            <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Warning /> Requirements
              </Typography>
              <Stack spacing={2}>
                {attraction.keyFacts?.min_height && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Height />
                    <Typography>
                      Minimum Height: {attraction.keyFacts.min_height}
                    </Typography>
                  </Box>
                )}
                {attraction.keyFacts?.min_age && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Person />
                    <Typography>
                      Minimum Age: {attraction.keyFacts.min_age}
                    </Typography>
                  </Box>
                )}
              </Stack>
            </Paper>
          </Slide>

          {/* Technical Stats Card */}
          <Slide direction="left" in timeout={1000}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Engineering /> Technical Stats
              </Typography>
              <Stack spacing={2}>
                {attraction.keyFacts?.topSpeed && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Speed />
                    <Typography>
                      Top Speed: {attraction.keyFacts.topSpeed}
                    </Typography>
                  </Box>
                )}
                {attraction.keyFacts?.duration && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Timer />
                    <Typography>
                      Duration: {attraction.keyFacts.duration}
                    </Typography>
                  </Box>
                )}
                {attraction.keyFacts?.manufacturer && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Engineering />
                    <Typography>
                      Manufacturer: {attraction.keyFacts.manufacturer}
                    </Typography>
                  </Box>
                )}
                {attraction.keyFacts?.opening_year && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CalendarToday />
                    <Typography>
                      Opening Year: {attraction.keyFacts.opening_year}
                    </Typography>
                  </Box>
                )}
              </Stack>
            </Paper>
          </Slide>
        </Grid>
      </Grid>
    </Container>
  );
};

export default AttractionPage; 