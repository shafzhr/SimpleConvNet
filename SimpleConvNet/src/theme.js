import { createMuiTheme } from '@material-ui/core/styles';


const theme = createMuiTheme({
    palette: {
      type: 'dark',
      background: {default: "#121212", paper: "rgba(255, 255, 255, 0.05)"},
      primary: {main: "#BB86FC"},
      secondary: {main: "#03DAC5"},
      error: {main: "#CF6679"}
    },
  });

export default theme
