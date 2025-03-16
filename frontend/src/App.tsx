import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material';
import { Box } from '@mui/material';
import theme from './theme';
import HomePage from './pages/HomePage';
import ParkInformationPage from './pages/ParkInformationPage';
import ParkAttractionsPage from './pages/ParkAttractionsPage';
import AttractionPage from './pages/AttractionPage';
import NavigationMenu from './components/NavigationMenu';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <NavigationMenu />
          <Box component="main" sx={{ flexGrow: 1 }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/park-info" element={<ParkInformationPage />} />
              <Route path="/attractions" element={<ParkAttractionsPage />} />
              <Route path="/attraction/:id" element={<AttractionPage />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
