import React from 'react'
import {Button} from "@material-ui/core"
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';


const theme = createMuiTheme({
  palette: {
    primary: {
      main: "#005582",
    },
    secondary: {
      main: '#11cb5f',
    },
  },
});


export function MyButton(props){


    return(
            <div style={{marginTop:20}}>
                <ThemeProvider theme={theme}>
                        <Button  variant="contained" color="primary" onClick={props.onClick} disabled={props.disabled}>{props.children}</Button>
                </ThemeProvider>
            </div>
    )

}