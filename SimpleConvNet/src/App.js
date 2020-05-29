import React from 'react';
import './App.css';
import Album from './Album';
import { MuiThemeProvider } from '@material-ui/core/styles';
import theme from './theme'


function App() {
  return (
    <MuiThemeProvider theme={theme}>
      <div className="App">
          <Album/>
      </div>
    </MuiThemeProvider>
  );
}

export default App;
