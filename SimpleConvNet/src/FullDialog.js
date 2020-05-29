import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import ListItemText from '@material-ui/core/ListItemText';
import ListItem from '@material-ui/core/ListItem';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
import CloseIcon from '@material-ui/icons/Close';
import Slide from '@material-ui/core/Slide';
import ButtonBase from '@material-ui/core/ButtonBase';
import Axios from 'axios';

const useStyles = makeStyles((theme) => ({
  appBar: {
    position: 'relative',
    backgroundColor: theme.palette.background.paper,
    color: theme.palette.primary.main,
  },
  title: {
    marginLeft: theme.spacing(2),
    flex: 1,
    color: theme.palette.primary.main,
  },
  buttonBaseAction: {
    height: '100%',
    width: '100%',
    display: 'block',
    textAlign: 'initial',
  },
  backBackground: {
    backgroundColor: theme.palette.background.default,
  },
  frontBackground: {
    backgroundColor: theme.palette.background.paper,
  },
  uploadButton: {
    backgroundColor: theme.palette.secondary.main,
  },
  respImg: {
    maxWidth: '100%',
    height: 'auto',
    display: 'block',
  },
}));

const Transition = React.forwardRef(function Transition(props, ref) {
  return <Slide direction="up" ref={ref} {...props} />;
});

export default function FullScreenDialog(props) {
  const classes = useStyles();
  const [open, setOpen] = React.useState(false);
  const [list, setList] = React.useState([]);
  const [previews, setPreviews] = React.useState([]);
  
  const onImageChange = (event) => {
    if (event.target.files) {
      let reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target.result;
        setPreviews(previews => [...previews, e.target.result]);
      };

      const new_files = [...event.target.files];
      let output = [];
      for (let i = 0; i < new_files.length; i++){
        if (new_files[i]) {
          reader.readAsDataURL(new_files[i]);
          output.push(new_files[i]);
        }
      }
      setList(list => [...list, ...output]);
    }
  }

  const uploadFiles = () => {
    const formData = new FormData();
    for (let i = 0; i < list.length; i++) {
      formData.append(`image[${i}]`, list[i])
    }
    Axios.post('/upload', formData, {
    }).then(response => {
        let result = "";
        for (let [key, value] of Object.entries(response.data)) {
          result += value + ","
        }
        result = result.substring(0, result.length-1)
        console.log(result);
        alert(result);
        setList([]);
        setPreviews([]);
    });
    
  }

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <div>
      <ButtonBase className={classes.buttonBaseAction} onClick={handleClickOpen}>
        {props.children}
      </ButtonBase>
      {/* <div className={classes.backBackground}> */}
      <Dialog fullScreen open={open} onClose={handleClose} TransitionComponent={Transition}>
        <div className={classes.backBackground}>
        <div className={classes.frontBackground}>
        <AppBar className={classes.appBar}>
            <Toolbar>
                <IconButton edge="start" color="inherit" onClick={handleClose} aria-label="close">
                <CloseIcon />
                </IconButton>
                <Typography variant="h6" className={classes.title}>
                Upload Photo
                </Typography>
                <Button autoFocus variant="contained" color="primary" onClick={() => {
                    uploadFiles(); handleClose(); }}>
                send
                </Button>
            </Toolbar>
        </AppBar>
        </div>
        </div>
        <List>
          {previews.map(item => (
                            <div>
                            <ListItem button>
                              <img className={classes.respImg} id="target" src={item}/>
                            </ListItem>
                            <Divider/>
                            </div>
                            ))}
          <ListItem>
            <Button className={classes.uploadButton} variant="contained" component="label">
                Upload Photo
                <input onChange={onImageChange} type="file" style={{ display: "none" }}/>
            </Button>
          </ListItem>
        </List>
      </Dialog>
    </div>
  );
}
