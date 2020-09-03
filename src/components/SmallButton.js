import React from 'react'
import {Button} from "@material-ui/core"
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';


const theme2 = createMuiTheme({
  palette: {
    primary: {
      main: "#005582",
    },
    secondary: {
      main: '#11cb5f',
    },
  },
});


export function SmallButton(props){


    return(
        <div style={{marginTop:10}}>
            <ThemeProvider theme={theme2}>
                    <Button  variant="outlined" color="primary" onClick={props.onClick} disabled={props.disabled}>{props.children}</Button>
            </ThemeProvider>
        </div>
    )

}