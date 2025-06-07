import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Chip,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  IconButton,
  Card,
  CardContent,
  Fade,
  Slide,
  Zoom,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Divider,
  Button,
  Stack,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Rating,
  CircularProgress,
  Alert,
} from '@mui/material';

import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css'; // Import stylesheet

import {
  ArrowBack,
  CalendarToday,
  LocationOn,
  Person,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Star,
  Info,
  AccessTime,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  CheckCircle,
  Cancel,
  Refresh,
  Speed,
  Height,
  Timer,
  Warning,
  Category,
} from '@mui/icons-material';

// Import attraction data
import attractionsData from '../data/attractions.json';
// Import the real-time queue times hook
import { useQueueTimes } from '../hooks/useQueueTimes';
// Import the historical queue times hook
import { useHistoricalQueueTimes } from '../hooks/useHistoricalQueueTimes';

// Import Chart.js and react-chartjs-2 components
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js/auto'; // Import 'chart.js/auto' for automatic registration

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);


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

  // Use the real-time queue times hook
  const { queueTimes, loading: queueLoading, error: queueError, getWaitTimeForRide } = useQueueTimes();

  // State for historical data
  const [selectedDate, setSelectedDate] = useState<Date | null>(new Date()); // Use Date object for react-datepicker

  // Use the historical queue times hook
  const { historicalData, loading: loadingHistoricalData, error: historicalError } = useHistoricalQueueTimes(id, selectedDate);

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

  // Get real-time wait time data for this attraction
  const rideWaitTime = getWaitTimeForRide(attraction.id);


  const formatWaitTime = () => {
    if (queueLoading) {
      return <CircularProgress size={20} />;
    }

    if (queueError) {
      return 'Error';
    }

    if (!rideWaitTime) {
      return 'N/A';
    }

    if (!rideWaitTime.isOpen) {
      return 'Closed';
    }

    return rideWaitTime.waitTime === 0 ? 'No Wait' : `${rideWaitTime.waitTime} min`;
  };

  const getWaitTimeColor = () => {
    if (!rideWaitTime || !rideWaitTime.isOpen) {
      return '#9e9e9e';
    }

    const waitTime = rideWaitTime.waitTime;
    if (waitTime === 0) return '#4caf50';
    if (waitTime <= 15) return '#8bc34a';
    if (waitTime <= 30) return '#ff9800';
    if (waitTime <= 60) return '#ff5722';
    return '#f44336';
  };

  const formatLastUpdated = () => {
    if (!rideWaitTime?.lastUpdated) return null;

    const updatedTime = new Date(rideWaitTime.lastUpdated);
    const now = new Date();
    const diffMinutes = Math.floor((now.getTime() - updatedTime.getTime()) / (1000 * 60));

    if (diffMinutes < 1) return 'Just updated';
    if (diffMinutes === 1) return '1 minute ago';
    if (diffMinutes < 60) return `${diffMinutes} minutes ago`;

    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours === 1) return '1 hour ago';
    return `${diffHours} hours ago`;
  };

   // Chart data configuration
   const chartData = {
    labels: historicalData?.map(data => {
      // Assuming timestamp is a Firestore Timestamp or similar object with a toDate() method
      if (data.timestamp && typeof data.timestamp.toDate === 'function') {
        return data.timestamp.toDate().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      }
      // Fallback for plain Date objects or other formats
      if (data.timestamp instanceof Date) {
        return data.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      }
      return ''; // Or handle other timestamp formats
    }) || [],
    datasets: [{
      label: 'Wait Time (min)',
      data: historicalData?.map(data => data.wait_time) || [],
      fill: false,
      borderColor: '#3f51b5', // Using a consistent color
      tension: 0.1,
    }],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false, // Allow the chart to adjust height
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `Historical Wait Times for ${attraction?.name || ''} on ${selectedDate?.toLocaleDateString() || ''}`,
      },
    },
     scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Wait Time (min)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time of Day'
        }
      }
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
            {rideWaitTime && (
              <Chip
                label={rideWaitTime.isOpen ? 'OPEN' : 'CLOSED'}
                sx={{
                  bgcolor: rideWaitTime.isOpen ? '#4caf50' : '#f44336',
                  color: 'white',
                  fontWeight: 'bold',
                }}
                icon={rideWaitTime.isOpen ?
                  <CheckCircle sx={{ color: 'white' }} /> :
                  <Cancel sx={{ color: 'white' }} />
                }
              />
            )}
          </Stack>
        </Slide>
      </Box>

      {/* Queue Times Error Alert */}
      {queueError && (
        <Alert severity="warning" sx={{ mb: 4 }}>
          <Typography variant="body2">
            Unable to load real-time wait times. Showing static information only.
          </Typography>
        </Alert>
      )}

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
                {queueLoading && <CircularProgress size={16} />}
              </Typography>
              <Typography
                variant="h3"
                sx={{
                  color: getWaitTimeColor(),
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}
              >
                {formatWaitTime()}
              </Typography>
              {rideWaitTime && rideWaitTime.isOpen && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {formatLastUpdated()}
                </Typography>
              )}
              {!queueLoading && !queueError && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
                  <Refresh sx={{ fontSize: 12 }} />
                  Updates every 5 minutes
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
                Technical Stats
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

        {/* Historical Data Section */}
        <Grid item xs={12}>
          <Fade in timeout={1000}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>Historical Wait Times</Typography>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom>Select a Date:</Typography>
                <DatePicker
                  selected={selectedDate}
                  onChange={(date: Date | null) => setSelectedDate(date)}
                  dateFormat="yyyy/MM/dd"
                  isClearable
                  placeholderText="Select a date"
                  customInput={<input style={{ width: '100%', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }} />}
                />
              </Box>

              {loadingHistoricalData && (
                <Box sx={{ textAlign: 'center', mt: 2 }}>
                  <CircularProgress />
                  <Typography>Loading historical data...</Typography>
                </Box>
              )}

              {!loadingHistoricalData && historicalError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Error loading historical data: {historicalError.message}
                  </Typography>
                </Alert>
              )}

              {!loadingHistoricalData && !historicalError && historicalData && historicalData.length > 0 ? (
                <Box sx={{ height: 400 }}> {/* Set a fixed height for the chart container */}
                  <Line data={chartData} options={chartOptions} />
                </Box>
              ) : !loadingHistoricalData && !historicalError && (
                <Typography variant="body1" sx={{ textAlign: 'center', mt: 2 }}>
                  No historical data available for the selected date.
                </Typography>
              )}
            </Paper>
          </Fade>
        </Grid>

      </Grid>
    </Container>
  );
};

export default AttractionPage;
