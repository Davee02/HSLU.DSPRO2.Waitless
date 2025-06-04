import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  InputAdornment,
  Chip,
  Stack,
  Divider,
  CardActionArea,
  Paper,
  Fab,
  Dialog,
  DialogContent,
  IconButton,
  useMediaQuery,
  useTheme,
  Button,
} from '@mui/material';
import { Search, Place, Speed, Height, Timer, // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Group, Event, Close, FilterList } from '@mui/icons-material';
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
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [selectedCategory, setSelectedCategory] = useState<Category>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredAttractions, setFilteredAttractions] = useState<Attraction[]>([]);
  const [isSearchOpen, setIsSearchOpen] = useState(false);

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

  const CategoryContent = () => (
    <Box 
      sx={{ 
        display: 'flex',
        justifyContent: 'center'
      }}
    >
      <Stack 
        direction="row" 
        spacing={1} 
        sx={{ 
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: 1,
          maxWidth: '100%',
          '& .MuiPaper-root': {
            minWidth: { xs: '120px', sm: '140px' },
            textAlign: 'center'
          }
        }}
      >
        {categories.map((category) => (
          <Paper
            key={category}
            elevation={selectedCategory === category ? 3 : 1}
            onClick={() => {
              setSelectedCategory(category);
              if (isMobile) setIsSearchOpen(false);
            }}
            sx={{
              px: 2,
              py: 1,
              cursor: 'pointer',
              backgroundColor: selectedCategory === category ? 'primary.main' : 'background.paper',
              color: selectedCategory === category ? 'black' : 'text.primary',
              fontWeight: selectedCategory === category ? 'bold' : 'normal',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                backgroundColor: selectedCategory === category ? 'primary.dark' : 'action.hover',
              },
            }}
          >
            <Typography
              sx={{
                textTransform: 'capitalize',
                whiteSpace: 'nowrap',
              }}
            >
              {category}
            </Typography>
          </Paper>
        ))}
      </Stack>
    </Box>
  );

  return (
    <Box sx={{ width: '100%' }}>
      <Typography 
        variant="h3" 
        component="h1" 
        gutterBottom 
        align="center" 
        sx={{ 
          mb: { xs: 2, md: 4 },
          fontSize: { xs: '1.75rem', sm: '2.25rem', md: '3rem' }
        }}
      >
        Park Attractions
      </Typography>

      {/* Search Bar - Always visible */}
      <Box sx={{ mb: { xs: 2, md: 4 } }}>
        <Grid 
          container 
          spacing={{ xs: 1, sm: 2, md: 3 }} 
          justifyContent="center" 
          alignItems="center"
        >
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
          {isMobile && (
            <Grid item xs="auto">
              <Button
                variant="contained"
                startIcon={<FilterList />}
                onClick={() => setIsSearchOpen(true)}
                sx={{
                  height: '56px',
                  px: 2,
                }}
              >
                Filter
              </Button>
            </Grid>
          )}
        </Grid>
      </Box>

      {/* Desktop Categories */}
      {!isMobile && (
        <Box sx={{ mb: { xs: 2, md: 4 } }}>
          <CategoryContent />
        </Box>
      )}

      {/* Mobile Category Dialog */}
      <Dialog
        fullScreen
        open={isMobile && isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        sx={{
          '& .MuiDialog-paper': {
            height: '100%',
            maxHeight: '100%',
          },
        }}
      >
        <DialogContent sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
            <IconButton onClick={() => setIsSearchOpen(false)}>
              <Close />
            </IconButton>
          </Box>
          <Typography variant="h6" gutterBottom>
            Filter by Category
          </Typography>
          <CategoryContent />
        </DialogContent>
      </Dialog>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
      >
        <Grid 
          container 
          spacing={{ xs: 2, sm: 3, md: 4 }}
          sx={{
            width: '100%',
            m: 0, // Remove default margin
          }}
        >
          {filteredAttractions.map((attraction) => (
            <Grid 
              item 
              xs={12} 
              sm={6} 
              md={4} 
              key={attraction.id}
              sx={{
                display: 'flex',
                width: '100%',
              }}
            >
              <motion.div variants={item} style={{ width: '100%' }}>
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
                          textTransform: 'capitalize',
                        }}
                      />
                    </Box>
                    <CardContent sx={{ flexGrow: 1 }}>
                      <Typography 
                        variant="h5" 
                        gutterBottom 
                        component="h2"
                        sx={{
                          fontSize: { xs: '1.25rem', sm: '1.5rem', md: '1.75rem' }
                        }}
                      >
                        {attraction.name}
                      </Typography>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        <Place sx={{ verticalAlign: 'middle', mr: 0.5 }} fontSize="small" />
                        {attraction.area}
                      </Typography>
                      <Typography 
                        variant="body2" 
                        color="text.secondary" 
                        paragraph
                        sx={{
                          fontSize: { xs: '0.875rem', sm: '1rem' }
                        }}
                      >
                        {attraction.short_description}
                      </Typography>
                      <Divider sx={{ my: 2 }} />
                      <Stack 
                        direction="row" 
                        spacing={1} 
                        flexWrap="wrap" 
                        useFlexGap 
                        sx={{ 
                          gap: 1,
                          '& .MuiChip-root': {
                            fontSize: { xs: '0.75rem', sm: '0.875rem' }
                          }
                        }}
                      >
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
    </Box>
  );
};

export default ParkAttractionsPage; 