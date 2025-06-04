import React, { useState } from 'react';
import { Box, Stack, Chip } from '@mui/material';

const AttractionsPage: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const categories = ['Category 1', 'Category 2', 'Category 3'];

  return (
    <Box sx={{ mb: 4, display: 'flex', justifyContent: 'center' }}>
      <Stack 
        direction={{ xs: 'row' }} 
        spacing={1} 
        sx={{ 
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: 1,
          '& .MuiChip-root': {
            minWidth: '100px',
            '&.Mui-selected': {
              backgroundColor: '#000000',
              color: '#ffffff',
              fontWeight: 'bold',
              '&:hover': {
                backgroundColor: '#000000',
              }
            }
          }
        }}
      >
        <Chip
          label="All"
          onClick={() => setSelectedCategory(null)}
          color={selectedCategory === null ? "primary" : "default"}
          variant={selectedCategory === null ? "filled" : "outlined"}
          sx={{ 
            borderColor: selectedCategory === null ? '#000000' : 'rgba(0, 0, 0, 0.23)',
            color: selectedCategory === null ? '#ffffff' : '#000000',
            '&:hover': {
              backgroundColor: selectedCategory === null ? '#000000' : 'rgba(0, 0, 0, 0.04)',
            }
          }}
        />
        {categories.map((category) => (
          <Chip
            key={category}
            label={category}
            onClick={() => setSelectedCategory(category)}
            color={selectedCategory === category ? "primary" : "default"}
            variant={selectedCategory === category ? "filled" : "outlined"}
            sx={{ 
              borderColor: selectedCategory === category ? '#000000' : 'rgba(0, 0, 0, 0.23)',
              color: selectedCategory === category ? '#ffffff' : '#000000',
              '&:hover': {
                backgroundColor: selectedCategory === category ? '#000000' : 'rgba(0, 0, 0, 0.04)',
              }
            }}
          />
        ))}
      </Stack>
    </Box>
  );
};

export default AttractionsPage; 