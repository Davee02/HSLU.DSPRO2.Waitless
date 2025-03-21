import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Grid,
  Tabs,
  Tab,
  Card,
  CardContent,
  TextField,
  InputAdornment,
  Chip,
  Stack,
  Divider,
  CardActionArea,
} from '@mui/material';
import { Search, Place, Speed, Height, Timer, // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Group, Event } from '@mui/icons-material';
import { motion, Variants } from 'framer-motion';
import { Attraction } from '../types';

// Import attraction data
import attractionsJson from '../data/attractions.json';

// Helper function to get correct image URLs
const getImageUrl = (imageName: string) => {
  try {
    return require(`../img/attractions/${imageName}`);
  } catch {
    return undefined;
  }
};

const categories = ['all', 'thrill', 'family', 'children', 'water', 'interactive'] as const;
type Category = typeof categories[number];

const ParkAttractionsPage: React.FC = () => {
  const navigate = useNavigate();
  const [selectedCategory, setSelectedCategory] = useState<Category>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredAttractions, setFilteredAttractions] = useState<Attraction[]>([]);

  const handleAttractionClick = (attractionId: string) => {
    navigate(`/attraction/${attractionId}`);
  };

  useEffect(() => {
    const attractions = (attractionsJson.attractions as unknown as Attraction[]).map(attraction => ({
      ...attraction,
      imageUrl: getImageUrl(attraction.imageUrl.split('/').pop() || '') || attraction.imageUrl
    }));
    
    const filtered = attractions.filter((attraction) => {
      const matchesCategory = selectedCategory === 'all' || attraction.category === selectedCategory;
      const matchesSearch = attraction.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          attraction.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          attraction.area.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    });
    setFilteredAttractions(filtered);
  }, [selectedCategory, searchQuery]);

  const container: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
        Park Attractions
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Search attractions, descriptions, or areas..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
        </Grid>
      </Box>

      <Tabs
        value={selectedCategory}
        onChange={(_, newValue: Category) => setSelectedCategory(newValue)}
        centered
        sx={{ mb: 4 }}
      >
        {categories.map((category) => (
          <Tab
            key={category}
            label={category.charAt(0).toUpperCase() + category.slice(1)}
            value={category}
          />
        ))}
      </Tabs>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
      >
        <Grid container spacing={3}>
          {filteredAttractions.map((attraction) => (
            <Grid item xs={12} md={6} lg={4} key={attraction.id}>
              <motion.div variants={item}>
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    transition: 'transform 0.2s, box-shadow 0.2s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: (theme) => theme.shadows[8],
                      cursor: 'pointer',
                    },
                  }}
                  onClick={() => handleAttractionClick(attraction.id)}
                >
                  <CardActionArea>
                    <Box
                      sx={{
                        position: 'relative',
                        paddingTop: '56.25%', // 16:9 aspect ratio
                        overflow: 'hidden',
                      }}
                    >
                      <Box
                        component="img"
                        src={attraction.imageUrl}
                        alt={attraction.name}
                        sx={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                          transition: 'transform 0.3s ease-in-out',
                          '&:hover': {
                            transform: 'scale(1.05)',
                          },
                        }}
                      />
                      <Chip
                        label={attraction.category}
                        sx={{
                          position: 'absolute',
                          top: 16,
                          right: 16,
                          backgroundColor: 'rgba(0, 0, 0, 0.6)',
                          color: 'white',
                        }}
                      />
                    </Box>
                    <CardContent sx={{ flexGrow: 1 }}>
                      <Typography variant="h5" gutterBottom component="h2">
                        {attraction.name}
                      </Typography>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        <Place sx={{ verticalAlign: 'middle', mr: 0.5 }} fontSize="small" />
                        {attraction.area}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {attraction.short_description}
                      </Typography>
                      <Divider sx={{ my: 2 }} />
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ gap: 1 }}>
                        {attraction.keyFacts.type && (
                          <Chip
                            size="small"
                            icon={<Speed />}
                            label={attraction.keyFacts.type}
                          />
                        )}
                        {attraction.keyFacts.duration && (
                          <Chip
                            size="small"
                            icon={<Timer />}
                            label={attraction.keyFacts.duration}
                          />
                        )}
                        {attraction.keyFacts.min_height && (
                          <Chip
                            size="small"
                            icon={<Height />}
                            label={attraction.keyFacts.min_height}
                          />
                        )}
                        {attraction.keyFacts.opening_year && (
                          <Chip
                            size="small"
                            icon={<Event />}
                            label={attraction.keyFacts.opening_year}
                          />
                        )}
                      </Stack>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </motion.div>
    </Container>
  );
};

export default ParkAttractionsPage; 