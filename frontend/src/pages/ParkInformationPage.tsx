import React, { useState } from 'react';
import 'flag-icons/css/flag-icons.min.css';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  Chip,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Paper,
  CardActionArea,
  useTheme,
  Fade,
  Slide,
  Zoom,
  IconButton,
  Tooltip,
  Link,
  Modal,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
} from '@mui/lab';
import {
  AccessTime,
  LocationOn,
  Euro,
  Hotel,
  ExpandMore,
  DirectionsCar,
  Train,
  Flight,
  Star,
  StarHalf,
  AttractionsOutlined,
  TheaterComedy,
  Pool,
  Celebration,
  Restaurant,
  ShoppingBag,
  Info,
  Accessible,
  Pool as WaterPark,
  Event,
  Park as Eco,
  Phone,
  Email,
  Facebook,
  Instagram,
  Twitter,
  YouTube,
  EmojiEvents,
  History,
  Store,
  LocalHospital,
  ChildCare,
  LocalAtm,
  Lock,
  AccessibleForward as Wheelchair,
  Wifi,
  Close as CloseIcon,
  LocalActivity,
  DinnerDining,
  Lightbulb,
  NavigateNext,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

// Import park data
import parkData from '../data/europapark.json';

// Import themed areas data
import themedAreasData from '../data/themed_areas.json';

const ParkInformationPage: React.FC = () => {
  const theme = useTheme();
  const [selectedArea, setSelectedArea] = useState<any>(null);
  const navigate = useNavigate();

  const handleAreaClick = (area: any) => {
    setSelectedArea(area);
  };

  const handleCloseModal = () => {
    setSelectedArea(null);
  };

  const handleAttractionClick = (attraction: any) => {
    navigate(`/attraction/${attraction.id}`);
  };

  const getHotelLink = (hotelName: string): string => {
    const hotelLinks: { [key: string]: string } = {
      'Hotel Colosseo': 'https://www.europapark.de/de/uebernachten/hotel-4-sterne-superior-erlebnishotel-colosseo',
      'Hotel Bell Rock': 'https://www.europapark.de/de/uebernachten/hotel-4-sterne-superior-erlebnishotel-bell-rock',
      'Hotel Krønasår': 'https://www.europapark.de/de/uebernachten/hotel-4-sterne-superior-erlebnishotel-kronasar',
      'Hotel Santa Isabel': 'https://www.europapark.de/en/accommodation/hotel-4-star-superior-santa-isabel',
      'Hotel El Andaluz': 'https://www.europapark.de/de/uebernachten/hotel-4-sterne-erlebnishotel-el-andaluz',
      'Hotel Castillo Alcazar': 'https://www.europapark.de/en/accommodation/hotel-4-star-castillo-alcazar'
    };
    return hotelLinks[hotelName] || '#';
  };

  const renderStars = (rating: number) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;

    for (let i = 0; i < fullStars; i++) {
      stars.push(<Star key={i} sx={{ color: '#000000' }} />);
    }
    if (hasHalfStar) {
      stars.push(<StarHalf key="half" sx={{ color: '#000000' }} />);
    }
    return stars;
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Park Overview */}
      <Box sx={{ mb: 8, textAlign: 'center' }}>
        <Typography 
          variant="h2" 
          component="h1" 
          gutterBottom 
          sx={{ 
            fontWeight: 'bold',
            color: '#000000',
            mb: 3
          }}
        >
          {parkData.parkInfo.name}
        </Typography>
        <Typography 
          variant="h4" 
          sx={{ 
            color: '#000000',
            mb: 4,
            fontStyle: 'italic'
          }}
        >
          {parkData.parkInfo.tagline}
        </Typography>
        <Typography 
          variant="body1" 
          sx={{ 
            maxWidth: '900px', 
            mx: 'auto',
            fontSize: '1.1rem',
            lineHeight: 1.8,
            color: '#000000',
            mb: 4
          }}
        >
          {parkData.parkInfo.shortDescription}
        </Typography>
        <Typography 
          variant="body1" 
          sx={{ 
            maxWidth: '900px', 
            mx: 'auto',
            fontSize: '1.1rem',
            lineHeight: 1.8,
            color: '#000000'
          }}
        >
          {parkData.parkInfo.longDescription}
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={4} sx={{ mb: 8 }}>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={3} sx={{ p: 3, textAlign: 'center', height: '100%' }}>
            <AttractionsOutlined sx={{ fontSize: 40, color: '#000000', mb: 2 }} />
            <Typography variant="h4" gutterBottom color="black">{parkData.parkInfo.attractions.total}</Typography>
            <Typography variant="subtitle1" color="black">Total Attractions</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={3} sx={{ p: 3, textAlign: 'center', height: '100%' }}>
            <Pool sx={{ fontSize: 40, color: '#000000', mb: 2 }} />
            <Typography variant="h4" gutterBottom color="black">{parkData.parkInfo.attractions.waterRides}</Typography>
            <Typography variant="subtitle1" color="black">Water Rides</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={3} sx={{ p: 3, textAlign: 'center', height: '100%' }}>
            <TheaterComedy sx={{ fontSize: 40, color: '#000000', mb: 2 }} />
            <Typography variant="h4" gutterBottom color="black">{parkData.parkInfo.attractions.showVenues}</Typography>
            <Typography variant="subtitle1" color="black">Show Venues</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={3} sx={{ p: 3, textAlign: 'center', height: '100%' }}>
            <DirectionsCar sx={{ fontSize: 40, color: '#000000', mb: 2 }} />
            <Typography variant="h4" gutterBottom color="black">{parkData.parkInfo.attractions.darkRides}</Typography>
            <Typography variant="subtitle1" color="black">Dark Rides</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={3} sx={{ p: 3, textAlign: 'center', height: '100%' }}>
            <AttractionsOutlined sx={{ fontSize: 40, color: '#000000', mb: 2 }} />
            <Typography variant="h4" gutterBottom color="black">{parkData.parkInfo.attractions.rollerCoasters}</Typography>
            <Typography variant="subtitle1" color="black">Roller Coasters</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Paper elevation={3} sx={{ p: 3, textAlign: 'center', height: '100%' }}>
            <Celebration sx={{ fontSize: 40, color: '#000000', mb: 2 }} />
            <Typography variant="h4" gutterBottom color="black">{parkData.parkInfo.themedAreas.length}</Typography>
            <Typography variant="subtitle1" color="black">Themed Areas</Typography>
          </Paper>
        </Grid>
      </Grid>

      <Grid container spacing={4}>
        {/* Location Information */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#000000' }}>
                <LocationOn sx={{ color: '#000000' }} /> Location & Directions
              </Typography>
              <Typography variant="h6" sx={{ mb: 2 }}>
                {parkData.parkInfo.location.city}, {parkData.parkInfo.location.state}, {parkData.parkInfo.location.country}
              </Typography>
              <Typography variant="body1" paragraph sx={{ fontWeight: 'medium' }}>
                {parkData.parkInfo.location.address}
              </Typography>
              <Divider sx={{ my: 3 }} />
              <Typography variant="h6" gutterBottom sx={{ color: '#000000' }}>How to Reach Us</Typography>
              <List>
                <ListItem>
                  <ListItemText 
                    primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><DirectionsCar /> By Car</Box>}
                    secondary={<Typography variant="body2" sx={{ mt: 1 }}>{parkData.parkInfo.location.howToReach.byCar}</Typography>}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><Train /> By Public Transport</Box>}
                    secondary={<Typography variant="body2" sx={{ mt: 1 }}>{parkData.parkInfo.location.howToReach.byPublicTransport}</Typography>}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><Flight /> By Plane</Box>}
                    secondary={<Typography variant="body2" sx={{ mt: 1 }}>{parkData.parkInfo.location.howToReach.byPlane}</Typography>}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Opening Times & Prices */}
        <Grid item xs={12} md={6}>
          <Stack spacing={4}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#000000' }}>
                  <AccessTime sx={{ color: '#000000' }} /> Opening Times
                </Typography>
                <List>
                  <ListItem>
                    <ListItemText 
                      primary={<Typography variant="h6">Regular Season</Typography>}
                      secondary={
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          <strong>{parkData.parkInfo.openingTimes.regularSeason.period}</strong><br />
                          {parkData.parkInfo.openingTimes.regularSeason.hours}
                        </Typography>
                      }
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary={<Typography variant="h6">Winter Season</Typography>}
                      secondary={
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          <strong>{parkData.parkInfo.openingTimes.winterSeason.period}</strong><br />
                          {parkData.parkInfo.openingTimes.winterSeason.hours}
                        </Typography>
                      }
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary={<Typography variant="h6">Closed Period</Typography>}
                      secondary={<Typography variant="body2" sx={{ mt: 1 }}>{parkData.parkInfo.openingTimes.closedPeriod}</Typography>}
                    />
                  </ListItem>
                </List>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                  {parkData.parkInfo.openingTimes.note}
                </Typography>
              </CardContent>
            </Card>

            <Card elevation={3}>
              <CardContent>
                <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#000000' }}>
                  <Euro sx={{ color: '#000000' }} /> Ticket Prices
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="h6" gutterBottom>Single Day</Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Adult"
                          secondary={parkData.parkInfo.prices.singleDay.adult}
                          secondaryTypographyProps={{ sx: { fontSize: '1.1rem', fontWeight: 'medium' } }}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Child"
                          secondary={parkData.parkInfo.prices.singleDay.child}
                          secondaryTypographyProps={{ sx: { fontSize: '1.1rem', fontWeight: 'medium' } }}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Senior"
                          secondary={parkData.parkInfo.prices.singleDay.senior}
                          secondaryTypographyProps={{ sx: { fontSize: '1.1rem', fontWeight: 'medium' } }}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="h6" gutterBottom>Season Pass</Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Standard"
                          secondary={parkData.parkInfo.prices.seasonPass.standard}
                          secondaryTypographyProps={{ sx: { fontSize: '1.1rem', fontWeight: 'medium' } }}
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Premium"
                          secondary={parkData.parkInfo.prices.seasonPass.premium}
                          secondaryTypographyProps={{ sx: { fontSize: '1.1rem', fontWeight: 'medium' } }}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom>Special Offers</Typography>
                <List dense>
                  {parkData.parkInfo.prices.specialOffers.map((offer, index) => (
                    <ListItem key={index}>
                      <ListItemText 
                        primary={offer.name}
                        secondary={offer.description}
                        primaryTypographyProps={{ sx: { fontWeight: 'medium' } }}
                      />
                    </ListItem>
                  ))}
                </List>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                  {parkData.parkInfo.prices.priceNote}
                </Typography>
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        {/* Themed Areas */}
        <Grid item xs={12}>
          <Typography variant="h4" gutterBottom sx={{ mt: 4, mb: 3, color: '#000000', display: 'flex', alignItems: 'center', gap: 1 }}>
            Themed Areas
          </Typography>
          <Grid container spacing={2}>
            {themedAreasData.themedAreas.map((area, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card elevation={3}>
                  <CardActionArea onClick={() => handleAreaClick(area)}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom color="black">
                        {area.name}
                      </Typography>
                      <Typography variant="body2" color="black">
                        {area.description.substring(0, 150)}...
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Detailed Area Modal */}
          <Dialog
            open={Boolean(selectedArea)}
            onClose={handleCloseModal}
            maxWidth="md"
            fullWidth
            scroll="paper"
            PaperProps={{
              sx: {
                borderRadius: 2,
                bgcolor: '#ffffff',
              }
            }}
          >
            {selectedArea && (
              <>
                <DialogTitle
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    bgcolor: '#ffffff',
                    color: '#000000',
                    py: 2,
                  }}
                >
                  <Typography variant="h5" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#000000' }}>
                    <span className={`fi fi-${selectedArea.countryCode?.toLowerCase()}`} style={{ fontSize: '1.5em' }} /> {selectedArea.name}
                  </Typography>
                  <IconButton
                    edge="end"
                    onClick={handleCloseModal}
                    aria-label="close"
                    sx={{ color: '#000000' }}
                  >
                    <CloseIcon />
                  </IconButton>
                </DialogTitle>
                <DialogContent dividers sx={{ p: 3 }}>
                  <Stack spacing={4}>
                    {/* Description */}
                    <Box>
                      <Typography variant="body1" sx={{ lineHeight: 1.8, color: '#000000' }}>
                        {selectedArea.description}
                      </Typography>
                    </Box>

                    {/* Attractions */}
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        color: '#000000',
                        borderBottom: 1,
                        borderColor: 'divider',
                        pb: 1
                      }}>
                        <LocalActivity sx={{ color: '#000000' }} /> Attractions
                      </Typography>
                      <Grid container spacing={2}>
                        {selectedArea.attractions.map((attraction: any, index: number) => (
                          <Grid item xs={12} sm={6} key={index}>
                            <Card 
                              variant="outlined" 
                              sx={{ 
                                height: '100%',
                                transition: 'transform 0.2s, box-shadow 0.2s',
                                '&:hover': {
                                  transform: 'translateY(-4px)',
                                  boxShadow: 4,
                                  cursor: 'pointer'
                                }
                              }}
                              onClick={() => handleAttractionClick(attraction)}
                            >
                              <CardContent>
                                <Stack spacing={1}>
                                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#000000' }}>
                                    {attraction.name}
                                  </Typography>
                                  <Chip
                                    icon={<AttractionsOutlined sx={{ color: '#000000' }} />}
                                    label={attraction.type}
                                    size="small"
                                    sx={{ 
                                      alignSelf: 'flex-start',
                                      color: '#000000',
                                      '& .MuiChip-label': { color: '#000000' },
                                      borderColor: '#000000'
                                    }}
                                    variant="outlined"
                                  />
                                  <Typography variant="body2" color="#000000">
                                    {attraction.description}
                                  </Typography>
                                </Stack>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>

                    {/* Dining */}
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        color: '#000000',
                        borderBottom: 1,
                        borderColor: 'divider',
                        pb: 1
                      }}>
                        <DinnerDining sx={{ color: '#000000' }} /> Dining
                      </Typography>
                      <Grid container spacing={2}>
                        {selectedArea.dining.map((restaurant: any, index: number) => (
                          <Grid item xs={12} sm={6} key={index}>
                            <Card 
                              variant="outlined"
                              sx={{ 
                                height: '100%',
                                transition: 'transform 0.2s, box-shadow 0.2s',
                                '&:hover': {
                                  transform: 'translateY(-4px)',
                                  boxShadow: 4,
                                }
                              }}
                            >
                              <CardContent>
                                <Stack spacing={1}>
                                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#000000' }}>
                                    {restaurant.name}
                                  </Typography>
                                  <Stack direction="row" spacing={1}>
                                    <Chip
                                      icon={<Restaurant sx={{ color: '#000000' }} />}
                                      label={restaurant.type}
                                      size="small"
                                      sx={{ 
                                        color: '#000000',
                                        '& .MuiChip-label': { color: '#000000' },
                                        borderColor: '#000000'
                                      }}
                                      variant="outlined"
                                    />
                                    <Chip
                                      label={restaurant.cuisine}
                                      size="small"
                                      sx={{ 
                                        color: '#000000',
                                        '& .MuiChip-label': { color: '#000000' },
                                        borderColor: '#000000'
                                      }}
                                      variant="outlined"
                                    />
                                  </Stack>
                                  <Typography variant="body2" color="#000000">
                                    {restaurant.description}
                                  </Typography>
                                </Stack>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>

                    {/* Shopping */}
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        color: '#000000',
                        borderBottom: 1,
                        borderColor: 'divider',
                        pb: 1
                      }}>
                        <Store sx={{ color: '#000000' }} /> Shopping
                      </Typography>
                      <Grid container spacing={2}>
                        {selectedArea.shopping.map((shop: any, index: number) => (
                          <Grid item xs={12} sm={6} key={index}>
                            <Card 
                              variant="outlined"
                              sx={{ 
                                height: '100%',
                                transition: 'transform 0.2s, box-shadow 0.2s',
                                '&:hover': {
                                  transform: 'translateY(-4px)',
                                  boxShadow: 4,
                                }
                              }}
                            >
                              <CardContent>
                                <Stack spacing={1}>
                                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#000000' }}>
                                    {shop.name}
                                  </Typography>
                                  <Chip
                                    icon={<ShoppingBag sx={{ color: '#000000' }} />}
                                    label={shop.type}
                                    size="small"
                                    sx={{ 
                                      alignSelf: 'flex-start',
                                      color: '#000000',
                                      '& .MuiChip-label': { color: '#000000' },
                                      borderColor: '#000000'
                                    }}
                                    variant="outlined"
                                  />
                                  <Typography variant="body2" color="#000000">
                                    {shop.description}
                                  </Typography>
                                </Stack>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>

                    {/* Events */}
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        color: '#000000',
                        borderBottom: 1,
                        borderColor: 'divider',
                        pb: 1
                      }}>
                        <Event sx={{ color: '#000000' }} /> Special Events
                      </Typography>
                      <Grid container spacing={2}>
                        {selectedArea.events.map((event: string, index: number) => (
                          <Grid item xs={12} sm={6} key={index}>
                            <Paper 
                              elevation={0} 
                              variant="outlined"
                              sx={{ 
                                p: 2,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1,
                                transition: 'transform 0.2s, box-shadow 0.2s',
                                '&:hover': {
                                  transform: 'translateY(-2px)',
                                  boxShadow: 2,
                                }
                              }}
                            >
                              <Event sx={{ color: '#000000' }} />
                              <Typography variant="body1" color="#000000">
                                {event}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>

                    {/* Fun Facts */}
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1,
                        color: '#000000',
                        borderBottom: 1,
                        borderColor: 'divider',
                        pb: 1
                      }}>
                        <Lightbulb sx={{ color: '#000000' }} /> Fun Facts
                      </Typography>
                      <Grid container spacing={2}>
                        {selectedArea.funFacts.map((fact: string, index: number) => (
                          <Grid item xs={12} key={index}>
                            <Paper 
                              elevation={0}
                              variant="outlined"
                              sx={{ 
                                p: 2,
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: 1,
                                bgcolor: 'primary.50',
                                transition: 'transform 0.2s',
                                '&:hover': {
                                  transform: 'translateX(8px)',
                                }
                              }}
                            >
                              <NavigateNext sx={{ mt: 0.5, color: '#000000' }} />
                              <Typography variant="body1" color="#000000">
                                {fact}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                  </Stack>
                </DialogContent>
                <DialogActions sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Button 
                    onClick={handleCloseModal}
                    variant="contained"
                    startIcon={<CloseIcon />}
                  >
                    Close
                  </Button>
                </DialogActions>
              </>
            )}
          </Dialog>
        </Grid>

        {/* Accommodation */}
        <Grid item xs={12}>
          <Typography variant="h4" gutterBottom sx={{ mt: 6, mb: 3, color: '#000000', display: 'flex', alignItems: 'center', gap: 1 }}>
            <Hotel sx={{ color: '#000000' }} /> Accommodation
          </Typography>
          <Grid container spacing={3}>
            {parkData.parkInfo.accommodation.hotels.map((hotel, index) => (
              <Grid item xs={12} md={6} lg={3} key={index}>
                <Card elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <CardActionArea 
                    component="a" 
                    href={getHotelLink(hotel.name)}
                    target="_blank"
                    sx={{ flexGrow: 1 }}
                  >
                    <CardContent>
                      <Typography variant="h6" gutterBottom color="black">
                        {hotel.name}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        {renderStars(hotel.stars)}
                      </Box>
                      <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                        <Chip label={hotel.theme} size="small" color="secondary" />
                        <Chip label={`${hotel.rooms} Rooms`} size="small" variant="outlined" />
                      </Stack>
                      <Typography variant="body2" paragraph>
                        {hotel.description}
                      </Typography>
                      <Divider sx={{ my: 1 }} />
                      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>Amenities:</Typography>
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ gap: 1 }}>
                        {hotel.amenities.map((amenity, i) => (
                          <Chip key={i} label={amenity} size="small" variant="outlined" />
                        ))}
                      </Stack>
                      <Typography variant="subtitle1" sx={{ mt: 2, color: '#000000', fontWeight: 'medium' }}>
                        {hotel.priceRange}
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Additional Information */}
        <Grid item xs={12} sx={{ mt: 6 }}>
          <Typography variant="h4" gutterBottom sx={{ mb: 3, color: '#000000' }}>
            Attraction Categories
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(parkData.parkInfo.attractions.byCategory).map(([category, count]) => (
              <Grid item xs={12} sm={6} md={4} key={category}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom color="black" sx={{ textTransform: 'capitalize' }}>
                      {category} Attractions
                    </Typography>
                    <Typography variant="h3" color="black">
                      {count}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Highlight Attractions */}
        <Grid item xs={12} sx={{ mt: 6 }}>
          <Typography variant="h4" gutterBottom sx={{ mb: 3, color: '#000000' }}>
            Highlight Attractions
          </Typography>
          <Grid container spacing={2}>
            {parkData.parkInfo.attractions.highlightAttractions.map((attraction, index) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" color="black">
                      {attraction}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>

      {/* Facilities Section */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Typography variant="h4" gutterBottom sx={{ color: '#000000', mb: 4 }}>
            Park Facilities
          </Typography>
        </Fade>
        
        {/* Dining */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Zoom in style={{ transitionDelay: '200ms' }}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Restaurant /> Dining Options
                  </Typography>
                  <Typography variant="h4" gutterBottom>
                    {parkData.parkInfo.facilities.dining.restaurants}
                  </Typography>
                  <Typography variant="subtitle1" gutterBottom>
                    Restaurants & Food Outlets
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ gap: 1, mt: 2 }}>
                    {parkData.parkInfo.facilities.dining.cuisine.map((cuisine, index) => (
                      <Chip key={index} label={cuisine} variant="outlined" />
                    ))}
                  </Stack>
                  <List>
                    {parkData.parkInfo.facilities.dining.highlights.map((restaurant, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={restaurant.name}
                          secondary={`${restaurant.type} - ${restaurant.cuisine}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Zoom>
          </Grid>

          {/* Shopping */}
          <Grid item xs={12} md={4}>
            <Zoom in style={{ transitionDelay: '400ms' }}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ShoppingBag /> Shopping
                  </Typography>
                  <Typography variant="h4" gutterBottom>
                    {parkData.parkInfo.facilities.shopping.stores}
                  </Typography>
                  <Typography variant="subtitle1" gutterBottom>
                    Retail Locations
                  </Typography>
                  <List>
                    {parkData.parkInfo.facilities.shopping.mainShops.map((shop, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={shop.name}
                          secondary={`${shop.type} - ${shop.location}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Zoom>
          </Grid>

          {/* Services */}
          <Grid item xs={12} md={4}>
            <Zoom in style={{ transitionDelay: '600ms' }}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Info /> Guest Services
                  </Typography>
                  <Stack spacing={2}>
                    {parkData.parkInfo.facilities.services.map((service, index) => (
                      <Paper key={index} elevation={1} sx={{ p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {service.name === 'Guest Services' && <Info />}
                          {service.name === 'First Aid' && <LocalHospital />}
                          {service.name === 'Baby Care Center' && <ChildCare />}
                          {service.name === 'ATMs' && <LocalAtm />}
                          {service.name === 'Lockers' && <Lock />}
                          {service.name === 'Wheelchair and Stroller Rental' && <Wheelchair />}
                          {service.name === 'WiFi' && <Wifi />}
                          {service.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {service.location}
                        </Typography>
                        {service.services && (
                          <List dense>
                            {service.services.map((s, i) => (
                              <ListItem key={i}>
                                <ListItemText primary={s} />
                              </ListItem>
                            ))}
                          </List>
                        )}
                        {service.price && (
                          <Typography variant="body2" color="text.secondary">
                            Price: {service.price}
                          </Typography>
                        )}
                      </Paper>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            </Zoom>
          </Grid>
        </Grid>
      </Box>

      {/* Accessibility */}
      <Box sx={{ mt: 8 }}>
        <Slide direction="right" in timeout={1000}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Accessible /> Accessibility
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body1" paragraph>
                    {parkData.parkInfo.facilities.accessibility.wheelchairAccess}
                  </Typography>
                  <Typography variant="body1" paragraph>
                    {parkData.parkInfo.facilities.accessibility.accessibleAttractions}% of attractions are accessible
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <List>
                    {parkData.parkInfo.facilities.accessibility.assistiveServices.map((service, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={service} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Slide>
      </Box>

      {/* Additional Venues */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Typography variant="h4" gutterBottom sx={{ color: '#000000', mb: 4 }}>
            Additional Venues
          </Typography>
        </Fade>
        <Grid container spacing={3}>
          {parkData.parkInfo.additionalVenues.map((venue, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Zoom in style={{ transitionDelay: `${200 * index}ms` }}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {venue.type === 'Water Park' && <WaterPark />}
                      {venue.type === 'VR Experience' && <TheaterComedy />}
                      {venue.type === 'Event Venue' && <Event />}
                      {venue.type === 'Conference Venue' && <Event />}
                      {venue.name}
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary" gutterBottom>
                      {venue.type}
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {venue.description}
                    </Typography>
                    {venue.pricing && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Pricing:
                        </Typography>
                        {Object.entries(venue.pricing).map(([key, value]) => (
                          <Typography key={key} variant="body2">
                            {key}: {value}
                          </Typography>
                        ))}
                      </Box>
                    )}
                    {venue.website && (
                      <Link href={venue.website} target="_blank" rel="noopener noreferrer">
                        Visit Website
                      </Link>
                    )}
                  </CardContent>
                </Card>
              </Zoom>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Special Events */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Typography variant="h4" gutterBottom sx={{ color: '#000000', mb: 4 }}>
            Special Events
          </Typography>
        </Fade>
        <Grid container spacing={3}>
          {parkData.parkInfo.specialEvents.map((event, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Zoom in style={{ transitionDelay: `${200 * index}ms` }}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {event.name}
                    </Typography>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      {event.period}
                    </Typography>
                    <Typography variant="body1">
                      {event.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Zoom>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Sustainability */}
      <Box sx={{ mt: 8 }}>
        <Slide direction="left" in timeout={1000}>
          <Card elevation={3} sx={{ bgcolor: '#f5f5f5' }}>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Eco /> Sustainability Initiatives
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <List>
                    {parkData.parkInfo.sustainability.initiatives.map((initiative, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={initiative} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" gutterBottom>
                    Environmental Awards
                  </Typography>
                  <List>
                    {parkData.parkInfo.sustainability.awards.map((award, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={award} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Slide>
      </Box>

      {/* Contact Information */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                Contact Information
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Typography sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Phone /> {parkData.parkInfo.contactInfo.phone}
                          </Typography>
                        }
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Typography sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Email /> {parkData.parkInfo.contactInfo.email}
                          </Typography>
                        }
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Link href={parkData.parkInfo.contactInfo.websiteURL} target="_blank" rel="noopener noreferrer">
                            Visit our website
                          </Link>
                        }
                      />
                    </ListItem>
                  </List>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Stack direction="row" spacing={2} justifyContent="center">
                    <IconButton
                      component="a"
                      href={parkData.parkInfo.contactInfo.socialMedia.facebook}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Facebook />
                    </IconButton>
                    <IconButton
                      component="a"
                      href={parkData.parkInfo.contactInfo.socialMedia.instagram}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Instagram />
                    </IconButton>
                    <IconButton
                      component="a"
                      href={parkData.parkInfo.contactInfo.socialMedia.twitter}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Twitter />
                    </IconButton>
                    <IconButton
                      component="a"
                      href={parkData.parkInfo.contactInfo.socialMedia.youtube}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <YouTube />
                    </IconButton>
                  </Stack>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Fade>
      </Box>

      {/* Awards */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Typography variant="h4" gutterBottom sx={{ color: '#000000', mb: 4 }}>
            <EmojiEvents /> Awards & Recognition
          </Typography>
        </Fade>
        <Grid container spacing={3}>
          {parkData.parkInfo.awards.map((award, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Zoom in style={{ transitionDelay: `${200 * index}ms` }}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {award.name}
                    </Typography>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      by {award.organization}
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ gap: 1, mt: 2 }}>
                      {award.years.map((year, i) => (
                        <Chip key={i} label={year} variant="outlined" size="small" />
                      ))}
                    </Stack>
                  </CardContent>
                </Card>
              </Zoom>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* History Timeline */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Typography variant="h4" gutterBottom sx={{ color: '#000000', mb: 4, display: 'flex', alignItems: 'center', gap: 1 }}>
            <History /> Park History
          </Typography>
        </Fade>
        <Box sx={{ overflowX: 'auto', pb: 2 }}>
          <Timeline position="alternate" sx={{ minWidth: 800 }}>
            {parkData.parkInfo.history.milestones.map((milestone, index) => (
              <TimelineItem key={index}>
                <TimelineOppositeContent color="text.secondary">
                  {milestone.year}
                </TimelineOppositeContent>
                <TimelineSeparator>
                  <TimelineDot color="primary" />
                  <TimelineConnector />
                </TimelineSeparator>
                <TimelineContent>
                  <Zoom in style={{ transitionDelay: `${100 * index}ms` }}>
                    <Paper elevation={3} sx={{ p: 2 }}>
                      <Typography variant="body1">
                        {milestone.event}
                      </Typography>
                    </Paper>
                  </Zoom>
                </TimelineContent>
              </TimelineItem>
            ))}
          </Timeline>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          {parkData.parkInfo.history.ownershipInfo}
        </Typography>
      </Box>

      {/* Park Statistics */}
      <Box sx={{ mt: 8 }}>
        <Fade in timeout={1000}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Park Statistics
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom>
                    Annual Visitors
                  </Typography>
                  <Typography variant="body1">
                    {parkData.parkInfo.statistics.annualVisitors}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom>
                    Park Size
                  </Typography>
                  <Typography variant="body1">
                    {parkData.parkInfo.statistics.parkSize}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom>
                    Employees
                  </Typography>
                  <Typography variant="body1">
                    Peak Season: {parkData.parkInfo.statistics.employees.peakSeason}
                    <br />
                    Year Round: {parkData.parkInfo.statistics.employees.yearRound}
                  </Typography>
                </Grid>
              </Grid>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                {parkData.parkInfo.statistics.economicImpact}
              </Typography>
            </CardContent>
          </Card>
        </Fade>
      </Box>
    </Container>
  );
};

export default ParkInformationPage; 