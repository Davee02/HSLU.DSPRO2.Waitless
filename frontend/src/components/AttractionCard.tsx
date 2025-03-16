import React from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { AccessTime } from '@mui/icons-material';
import { Attraction } from '../types';

// Import attraction images
import silverStar from '../img/attractions/silver_star.jpg';
import blueFire from '../img/attractions/blue_fire.jpg';
import wodan from '../img/attractions/wodan.jpg';
import voletarium from '../img/attractions/voletarium.jpg';

// Map for image imports
const imageMap: { [key: string]: string } = {
  '/static/silver_star.jpg': silverStar,
  '/static/blue_fire.jpg': blueFire,
  '/static/wodan.jpg': wodan,
  '/static/voletarium.jpg': voletarium,
};

interface AttractionCardProps {
  attraction: Attraction;
  showDetails?: boolean;
}

const AttractionCard: React.FC<AttractionCardProps> = ({ attraction, showDetails = false }) => {
  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardMedia
        component="img"
        height="200"
        image={imageMap[attraction.imageUrl]}
        alt={attraction.name}
        sx={{ objectFit: 'cover' }}
      />
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography gutterBottom variant="h5" component="div">
          {attraction.name}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Chip
            icon={<AccessTime />}
            label={attraction.waitTime ? `${attraction.waitTime} min wait` : 'No wait time data'}
            color={attraction.waitTime && attraction.waitTime > 30 ? 'error' : 'success'}
          />
        </Box>

        <Typography variant="body2" color="text.secondary" paragraph>
          {attraction.description}
        </Typography>

        {showDetails && (
          <List dense>
            {Object.entries(attraction.keyFacts).map(([key, value]) => (
              <ListItem key={key}>
                <ListItemText
                  primary={value}
                  secondary={key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                />
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default AttractionCard; 