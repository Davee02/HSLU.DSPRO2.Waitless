import React from 'react';
import { useNavigate } from 'react-router-dom';
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
  CardActionArea,
  CircularProgress,
} from '@mui/material';
import { AccessTime } from '@mui/icons-material';
import { Attraction } from '../types';
import { useQueueTimes } from '../hooks/useQueueTimes';

interface AttractionCardProps {
  attraction: Attraction;
  showDetails?: boolean;
}

const AttractionCard: React.FC<AttractionCardProps> = ({ attraction, showDetails = false }) => {
  const navigate = useNavigate();
  const { getWaitTimeForRide, loading: queueLoading } = useQueueTimes();
  const rideWaitTime = getWaitTimeForRide(attraction.id);

  const handleClick = () => {
    navigate(`/attraction/${attraction.id}`);
  };

  const getWaitTimeColor = () => {
    if (!rideWaitTime || !rideWaitTime.isOpen) {
      return 'default';
    }
    
    const waitTime = rideWaitTime.waitTime;
    if (waitTime === 0) return 'success';
    if (waitTime <= 15) return 'success';
    if (waitTime <= 30) return 'warning';
    if (waitTime <= 60) return 'warning';
    return 'error';
  };

  const formatWaitTime = () => {
    if (queueLoading) {
      return <CircularProgress size={16} />;
    }
    
    if (!rideWaitTime) {
      return 'No wait time data';
    }
    
    if (!rideWaitTime.isOpen) {
      return 'Closed';
    }
    
    return rideWaitTime.waitTime === 0 ? 'No Wait' : `${rideWaitTime.waitTime} min`;
  };

  return (
    <Card 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: (theme) => theme.shadows[8],
        },
      }}
    >
      <CardActionArea onClick={handleClick}>
        <CardMedia
          component="img"
          height="200"
          image={attraction.imageUrl}
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
              label={formatWaitTime()}
              color={getWaitTimeColor()}
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
      </CardActionArea>
    </Card>
  );
};

export default AttractionCard; 