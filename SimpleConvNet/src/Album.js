import React from 'react';
import AppBar from '@material-ui/core/AppBar';
import Button from '@material-ui/core/Button';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import CssBaseline from '@material-ui/core/CssBaseline';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import IconButton from '@material-ui/core/IconButton';
import AddCircleIcon from '@material-ui/icons/AddCircle';
import StatDialogButton from './StatDialog'
import FullScreenDialog from './FullDialog'

const useStyles = makeStyles((theme) => ({
  icon: {
    marginRight: theme.spacing(2),
  },
  heroContent: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(8, 0, 6),
  },
  heroButtons: {
    marginTop: theme.spacing(4),
  },
  cardGrid: {
    paddingTop: theme.spacing(8),
    paddingBottom: theme.spacing(8),
  },
  card: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  cardMedia: {
    paddingTop: '56.25%', // 16:9
    width: '100%',
    height: '100%',
  },
  cardContent: {
    flexGrow: 1,
    width: '100%',
    height: '100%',
  },
  addPhotoIcon: {
    color: theme.palette.secondary.main,
    iconStyle: { height: 48, width: 48},
    position: "fixed",
    bottom: theme.spacing.unit * 2,
    right: theme.spacing.unit * 3,
    // padding: 24,
  },
  largeIcon: {
    fontSize: "3em"
  },
  cardAction: {
    display: 'block',
    textAlign: 'initial'
  },
}));

const cards = [1, 2, 3];

export default function Album() {
  const classes = useStyles();
  return (
    <React.Fragment>
      <CssBaseline />
      <main>
        {/* Hero unit */}
        <div className={classes.heroContent}>
          <Container maxWidth="sm">
            <Typography component="h1" variant="h2" align="center" color="textPrimary" gutterBottom>
              Image Classifier
            </Typography>
            <Typography variant="h5" align="center" color="textSecondary" paragraph>
              Some short paragraph about the project
            </Typography>
            <div className={classes.heroButtons}>
              <Grid container spacing={2} justify="center">
                <Grid item>
                  <Button target="_blank" href="https://github.com/shafzhr/SimpleConvNet" variant="contained" color="primary">
                    Source Code
                  </Button>
                </Grid>
                {/* <Grid item>
                  <Button variant="outlined" color="primary">
                    Secondary action
                  </Button>
                </Grid> */}
              </Grid>
            </div>
          </Container>
        </div>
        <Container className={classes.cardGrid} maxWidth="md">
          {/* End hero unit */}
          <Grid container spacing={4}>
            {cards.map((card) => (
              <Grid item key={card} xs={12} sm={6} md={4}>
                <StatDialogButton
                  left={
                  <Card className={classes.card}>
                    <CardMedia
                      className={classes.cardMedia}
                      image="https://source.unsplash.com/random"
                      title="Image title"
                    />
                    <CardContent className={classes.cardContent}>
                      <Typography gutterBottom variant="h5" component="h2">
                        Stats
                      </Typography>
                      <Typography>
                        Stats from Kibana
                      </Typography>
                    </CardContent>
                  </Card>
                  }
                  right={
                    <Typography>Hello Dialog</Typography>
                  }
                />
              </Grid>
            ))}
          </Grid>
        </Container>
        <label htmlFor="icon-button-file">
        <FullScreenDialog>
          <IconButton className={classes.addPhotoIcon} aria-label="upload picture" component="span">
            <AddCircleIcon className={classes.largeIcon}/>
          </IconButton>
        </FullScreenDialog>
        </label>
      </main>
    </React.Fragment>
  );
}