import { createTheme } from '@mui/material/styles';

// Define a modern, muted color palette
const colors = {
  sage: {
    50: '#f2f4f3',
    100: '#e6e9e7',
    200: '#ccd3cf',
    300: '#b3bdb7',
    400: '#99a79f',
    500: '#809187',
    600: '#667b6f',
    700: '#4d6557',
    800: '#334f3f',
    900: '#1a3927',
  },
  clay: {
    50: '#f7f4f2',
    100: '#efe9e5',
    200: '#dfd3cb',
    300: '#cfbdb1',
    400: '#bfa797',
    500: '#af917d',
    600: '#8f7b63',
    700: '#6f6549',
    800: '#4f4f2f',
    900: '#2f3915',
  },
  slate: {
    50: '#f2f3f4',
    100: '#e5e7e9',
    200: '#cbd0d3',
    300: '#b1b9bd',
    400: '#97a2a7',
    500: '#7d8b91',
    600: '#637477',
    700: '#495d5d',
    800: '#2f4643',
    900: '#152f29',
  }
};

// Create the theme
const theme = createTheme({
  palette: {
    primary: {
      main: colors.sage[600],
      light: colors.sage[400],
      dark: colors.sage[800],
      contrastText: '#ffffff',
    },
    secondary: {
      main: colors.clay[500],
      light: colors.clay[300],
      dark: colors.clay[700],
      contrastText: '#ffffff',
    },
    background: {
      default: colors.sage[50],
      paper: '#ffffff',
    },
    text: {
      primary: colors.slate[800],
      secondary: colors.slate[600],
    },
    divider: colors.sage[200],
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 600,
      color: colors.slate[800],
    },
    h6: {
      fontWeight: 600,
      color: colors.slate[700],
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          padding: '8px 16px',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
          },
        },
        outlined: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

export default theme; 