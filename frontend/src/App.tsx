import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme';
import HomePage from './pages/HomePage';
import ParkAttractionsPage from './pages/ParkAttractionsPage';
import ParkInformationPage from './pages/ParkInformationPage';
import NavigationMenu from './components/NavigationMenu';
import { Box } from '@mui/material';

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Box sx={{ 
          display: 'flex',
          minHeight: '100vh',
          backgroundColor: (theme) => theme.palette.background.default
        }}>
          <NavigationMenu />
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              pt: 2,
              pb: 2,
              pr: 2,
              width: { sm: `calc(100% - 240px)` }
            }}
          >
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/park-info" element={<ParkInformationPage />} />
              <Route path="/attractions" element={<ParkAttractionsPage />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;
