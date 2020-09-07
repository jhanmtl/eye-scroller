import React from 'react'
import {CircularProgress} from "@material-ui/core";
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import InputLabel from "@material-ui/core/InputLabel";



const theme = createMuiTheme({
  palette: {
    primary: {
      main: "#c6c6c6",
    },
  },

    typography:{
      fontSize:10,
    },
});

export function MyProgress(){
    return(
        <div style={{textAlign:"center", marginTop:64}}>
            <div>
                <ThemeProvider theme={theme}>
                    <CircularProgress />
                </ThemeProvider>
            </div>
            <br/>
            <InputLabel>
                initiating model ...
            </InputLabel>
        </div>
    )

}