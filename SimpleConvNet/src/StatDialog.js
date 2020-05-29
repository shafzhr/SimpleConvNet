import React from 'react';
import PropTypes from 'prop-types';
import { makeStyles } from '@material-ui/core/styles';
import DialogTitle from '@material-ui/core/DialogTitle';
import Dialog from '@material-ui/core/Dialog';
import ButtonBase from '@material-ui/core/ButtonBase';

const useStyles = makeStyles((theme) => ({
  buttonBaseAction: {
    height: '100%',
    width: '100%',
    display: 'block',
    textAlign: 'initial',
  },

}));

function StatDialog(props) {
    const onClose = props.onClose; 
    const open = props.open;
  
    const handleClose = () => {
      onClose();
    };
    
    return (
      <Dialog onClose={handleClose} aria-labelledby="simple-dialog-title" open={open}>
        {props.children}
      </Dialog>
    );
  }

StatDialog.propTypes = {
  onClose: PropTypes.func.isRequired,
  open: PropTypes.bool.isRequired,
};

export default function StatDialogButton(props) {
  const classes = useStyles();

  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <div>
      <ButtonBase className={classes.buttonBaseAction} onClick={handleClickOpen}>
        {props.left}
      </ButtonBase>
      <StatDialog open={open} onClose={handleClose}>
        {props.right}
      </StatDialog>
    </div>
  );
}
