import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, useMediaQuery } from '@mui/material';
import { Box } from '@mui/material';
import theme from './theme';
import HomePage from './pages/HomePage';
import ParkInformationPage from './pages/ParkInformationPage';
import ParkAttractionsPage from './pages/ParkAttractionsPage';
import AttractionPage from './pages/AttractionPage';
import NavigationMenu from './components/NavigationMenu';

function App() {
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <NavigationMenu />
          <Box 
            component="main" 
            sx={{ 
              flexGrow: 1,
              ml: { xs: 0, md: '240px' }, // No margin on mobile, margin on desktop
              width: { xs: '100%', md: `calc(100% - 240px)` }, // Full width on mobile, adjusted on desktop
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              overflowX: 'hidden', // Prevent horizontal scrolling
            }}
          >
            <Box
              sx={{
                width: '100%',
                px: { xs: 3, sm: 4, md: 5 }, // Increased padding for small gaps on sides
                py: { xs: 2, sm: 3, md: 4 },
                boxSizing: 'border-box', // Ensure padding is included in width
              }}
            >
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/park-info" element={<ParkInformationPage />} />
                <Route path="/attractions" element={<ParkAttractionsPage />} />
                <Route path="/attraction/:id" element={<AttractionPage />} />
              </Routes>
            </Box>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
