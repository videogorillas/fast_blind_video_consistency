Docker:

```bash
docker run --rm -ti -v /outer_path/:/input_path/ r.c.videogorillas.com/video_consistency:latest
python runFolder.py --input <path to to input NOT consistent files folder> --output <path to to output folder> --origin <path to origin consistent files folder> --first 1 --last 20
```

Kubernetes:

```bash
kubectl create namespace video_consistency
kubectl -n video_consistency create -f kube.yml
```