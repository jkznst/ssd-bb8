#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 evaluate.py --rec-path ./data/obj01/val.rec --network resnet50m --batch-size 48 --data-shape 300 --class-names 'obj_01' --prefix ./output/obj01/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 1

#python3 evaluate.py --rec-path ./data/obj02/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_02' --prefix ./output/obj02/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj04/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_04' --prefix ./output/obj04/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj05/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_05' --prefix ./output/obj05/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj06/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_06' --prefix ./output/obj06/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj08/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_08' --prefix ./output/obj08/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj09/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_09' --prefix ./output/obj09/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj10/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_10' --prefix ./output/obj10/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj11/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_11' --prefix ./output/obj11/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj12/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_12' --prefix ./output/obj12/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj13/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_13' --prefix ./output/obj13/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj14/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_14' --prefix ./output/obj14/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0

#python3 evaluate.py --rec-path ./data/obj15/val.rec --network resnet50m --batch-size 32 --data-shape 300 --class-names 'obj_15' --prefix ./output/obj15/resnet50-300-lr0.001-OHEM-alpha10-wd0.0005-200-nofirstpooling/ssd --gpu 0


