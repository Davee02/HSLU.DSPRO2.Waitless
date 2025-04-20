import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
  ListItemButton,
} from '@mui/material';
import { Home, Info, Attractions } from '@mui/icons-material';
import logo from '../img/logo.png';

const drawerWidth = 240;

const menuItems = [
  {
    text: 'Home',
    icon: <Home />,
    path: '/'
  },
  {
    text: 'Park Information',
    icon: <Info />,
    path: '/park-info'
  },
  {
    text: 'Park Attractions',
    icon: <Attractions />,
    path: '/attractions'
  }
];

const NavigationMenu: React.FC = () => {
  const theme = useTheme();
  const location = useLocation();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: theme.palette.primary.main,
          color: theme.palette.primary.contrastText,
        },
      }}
    >
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <img src={logo} alt="Logo" style={{ maxWidth: '80%', height: 'auto' }} />
      </Box>
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: theme.palette.primary.dark,
                  '&:hover': {
                    backgroundColor: theme.palette.primary.dark,
                  },
                },
                '&:hover': {
                  backgroundColor: theme.palette.primary.light,
                },
              }}
            >
              <ListItemIcon sx={{ color: theme.palette.primary.contrastText }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default NavigationMenu; 