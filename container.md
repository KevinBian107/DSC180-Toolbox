# [Environments and Containerization](/README.md)

***Notes copied from dsc [180A capstone website](https://dsc-capstone.org/2025-26/assignments/methodology/06/)***

## Docker

In the event that your project involves more than just Python packages, conda environments might not suffice. That’s where Docker comes in. Docker provides you with a sandbox, in which you can run applications and install packages without impacting other parts of your computer. Specifically:

- A **Docker image** is a snapshot of a development environment. An image contains everything you need to run an application – all dependencies, configuration, scripts, binaries, etc.

- A **container** is created using an image. Once you launch a container, you are in the aforementioned sandbox. A container is simply another process on your machine that has been isolated from all other processes on the host machine.

Images and containers are closely related, but they’re not the same thing. Here’s an analogy: when Apple released iOS 17, they released an image. When you updated your iPhone to iOS 17 and started using it, you used Apple’s iOS 17 image to launch a container. This is an imperfect analogy, because images are not the same thing as operating systems, but it should hopefully clarify how images and containers are different. [Read this article for more details](https://www.howtogeek.com/devops/where-are-docker-images-containers-stored-on-the-host/). We can even run multiple containers at a time on a single machine, each using the same or different images.

Each time when we use DataHub or run a launch script on DSMLP, we are indirectly specifing a Docker image that we’d like to launch. we’ve also done our work inside a Docker container! For instance, [here is the Dockerfile that specifies the image that is used when we run `launch-scipy-ml.sh` on DSMLP](https://github.com/ucsd-ets/datahub-docker-stack/blob/main/images/scipy-ml-notebook/Dockerfile).

## Container Registry and DSMLP

Docker can be accessed remotely, either from GitHub or DockerHub: one place to store Docker images is [GitHub Container Registry (GHCR)](https://github.com/features/actions), which is what we’ll use here, another place is [Docker Hub](http://hub.docker.com/). For their Docker images, DSMLP used to use Docker Hub, but they switched over to GHCR recently. For instance, ghcr.io/ucsd-ets/[datascience-notebook:2023.4-notebook7](datascience-notebook:2023.4-notebook7) is the base image that all other DataHub and DSMLP images are based off of.

When launching a container on DSMLP, we can specify an image from online that we’d like to launch our container in. For instance, suppose we want to launch a container using the [base data science image linked above](https://github.com/ucsd-ets/datahub-docker-stack/pkgs/container/datascience-notebook). To do so, we need to find the path for the image. Images on GHCR have paths of the form `ghcr.io/<user>/<image>:<tag>`. For this image, the user is `ucsdets`, the image is `datascience-notebook`, and the most recent tag is `2023.4-notebook7`, so we’ll use that. (Other tags are listed below; [read more about tags here](https://www.freecodecamp.org/news/an-introduction-to-docker-tags-9b5395636c2a/).)

To launch a container on DSMLP with a specified image, we run

```bash
launch.sh -i <image_path> -W DSC180A_FA25_A00
```

where -i stands for image. So for the purposes of this example, we’ll run

```bash
launch.sh -i ghcr.io/ucsd-ets/datascience-notebook:2023.4-notebook7 -W DSC180A_FA25_A00
```

Great! Upon launch, we’ll be placed in a container whose environment is specified by the image we used. In the future, when we create our own images and upload them to GHCR, we’ll launch servers on DSMLP with the path to our image, instead.

## Dockerfiles

Pre-existing images are great, but often we’ll need to create our own image, customized for our project. Images are designed to be layered. That is, we can start with an existing layer and specify what we want to add on top of that layer (packages, languages, files, etc). The existing layer that we started with was itself likely layered upon other pre-existing images. If we continue up the hierarchy, the very first layer that our layer inherits from must have specified an operating system to use – we will exclusively work with Linux distributions; Ubuntu is quite common. (Note that we don’t need a Linux computer to build a Linux image!)

How does one specify what components to include in an image? By writing a **Dockerfile**. A Dockerfile is a plain text file that contains a sequence of steps that one can use to recreate our development environment. To create an image, we build a Dockerfile. In keeping with the iOS 17 example from earlier, if Apple’s release version of iOS 17 is an image and the copy of it running on my iPhone is a container, then the Dockerfile is the source code Apple wrote to create iOS 17. [More about Dockerfile videos here](https://www.youtube.com/watch?v=YFl2mCHdv24&t=159s).

Starting with an easy example:

```bash
FROM ghcr.io/ucsd-ets/datascience-notebook:2023.4-notebook7

USER root

RUN conda install --quiet --yes geopandas
```

The first line specifies that we want to start by adopting the `ghcr.io/ucsd-ets/datascience-notebook:2023.4-notebook7` image that we looked at earlier. Everything that is part of `ghcr.io/ucsd-ets/datascience-notebook` will be included in our new image, too. The next line specifies that we want to run all subsequent commands as the root (administrator) on the computer. The next line installs `geopandas` using conda (Note: this requires conda to be installed in the image. If it’s not, this will error!). Similarly, a real life example can be the [docker file for dsc 10](https://github.com/ucsd-ets/dsc10-notebook/blob/main/Dockerfile).

```bash
ARG BASE_CONTAINER=ghcr.io/ucsd-ets/datascience-notebook:2023.4-stable
FROM $BASE_CONTAINER

USER root

RUN pip install coverage==5.5 && \
  pip install 'pandas>=0.24, <= 1.5.3' babypandas==0.1.9 pandas-tutor==2.0.3 && \
  pip install otter-grader==3.3.0 && \
  pip install wordcloud==1.8.1

USER $NB_UID
```

It’s starting from a base image maintained by UCSD ETS (though not the same one as we looked at above), and installing the Python packages needed for DSC 10.

While Dockerfiles can get quickly get complicated, note that we only need to specify what we want to include in our image on top of an existing image. we will almost always start by using one of the images pre-configured by UCSD ETS as a base image. For a full reference of [all the commands that Dockerfiles understand, look here](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#dockerfile-instructions).

Note: Dockerfiles must be named Dockerfile with no extension. Otherwise, Docker Desktop won’t recognize them when you try to build our image. Once we’ve created a Dockerfile, the next step is to actually build an image from that Dockerfile. We’ll walk through how to do that below, in the context of a real example.

## Walk Through Docker Setup

We will walk through the docker setup process by [following this video](https://www.youtube.com/watch?v=5Z-Cu4_SJg8) from dsc180a.

### Setup
We’ll need to install [Docker Desktop](https://www.docker.com/products/docker-desktop/), a client for macOS, Windows, and Linux, locally on the computer. While DSMLP works with Docker, it does not have Docker Desktop installed. Installing Docker and pulling existing images requires having 5-10GB of free space on the computer.

The video earlier mentioned something called `boot2docker`. It is now deprecated and has been replaced with Docker Desktop. When we install Docker Desktop and pull an image, we essentially install the ability to run Linux-based applications, whether or not we have a Linux personal computer.

### Creating a Dockerfile

First, create a folder on our computer. Within it create a file with no extensions named `Dockerfile`. In it, copy the [Dockerfile template from this tutorial link](https://github.com/ucsd-ets/datahub-example-notebook/blob/main/Dockerfile). Then, modify the Dockerfile so that:

- add `RUN apt update` before the aforementioned line.
- `aria2`, `nmap`, and `traceroute` are installed using `apt-get`.
- `geopandas` and `babypandas==0.1.5` are installed using `conda` or `pip`.

There is nothing special about these packages; just need to install something for testing!

### Creating an Image

To create an image using thhis Dockerfile, use the following command in our Terminal/command-prompt:

```
docker build -t dsc-example-docker-image .
```

Replace `dsc-example-docker-image` with an image name we choose; our image can be named anything, but it should be something informative to our project.

**Make sure Docker Desktop is running on the computer at the same time, otherwise this won’t work!** It may take >10 minutes to create the image for the first time. This is to be expected, because it’s pulling ~5GB of files. Subsequent pulls will be much faster.

### Testing the Image
Once the image is built, run the following command in Terminal:

```bash
docker run --rm -it dsc-example-docker-image
```

To verify that packages we installed via `apt-get` are installed, run `which <package-name>`, e.g. `which nmap` (for aria2, use which aria2c). To verify that packages we installed via `conda` or `pip` are installed, open a Python interpreter and try to import them, or use `pip show <package name>`.


### Push the Image to GHCR

Before pushing to GHCR, we’re going to need to tell the Docker client what the GitHub credentials are. Start by creating a [new (classic) personal access token that has permission to read, write, and delete packages](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens). Then, in Terminal run

```bash
export CR_PAT=9q30thpalagekaef
```

with `9q30thpalagekaef` replaced with our token (the above token is fake). Then, run

```bash
echo $CR_PAT | docker login ghcr.io -u KevinBian107 --password-stdin
```

Now, we’re ready to push our image to GHCR. Start by using docker tag to specify a path for our image on GHCR:

```bash
docker tag dsc-example-docker-image ghcr.io/KevinBian107/dsc-example-docker-image
```

Finally, execute:

```bash
docker push ghcr.io/ubellur/dsc-example-docker-image
```

This should take a few minutes, but after, we should be able to navigate to the GitHub profile, click “Packages” at the top, and see `dsc-example-docker-image` as a private package there. Click the package, click “Package Settings” on the right, and at the bottom, turn it into a public package. This will allow us to access it from DSMLP.

### Use the Image on DSMLP
Log onto the DSMLP jumpbox server. From there, run:

```bash
launch.sh -i ghcr.io/KevinBian107/dsc-example-docker-image -W DSC180A_FA25_A00
```