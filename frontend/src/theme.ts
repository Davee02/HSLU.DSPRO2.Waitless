import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#f0f0f0',
      light: '#ffffff',
      dark: '#e0e0e0',
      contrastText: '#333333'
    },
    secondary: {
      main: '#e0e0e0',
      light: '#f5f5f5',
      dark: '#cccccc',
      contrastText: '#333333'
    },
    background: {
      default: '#ffffff',
      paper: '#ffffff'
    }
  }
});

export default theme; 