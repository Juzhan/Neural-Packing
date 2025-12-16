#!/usr/bin/env python3
"""
Data generation script for TAP benchmark.
Generates box data for three data types: rand, fix, ppsg.
Uses default parameters from main.py.
"""

import numpy as np
import os
import argparse
from tap.envs.factory import Factory
from tap.envs.container import Container
import tqdm

def get_default_args():
    """Get default arguments from main.py"""
    class Args:
        def __init__(self):
            self.container_size = [100, 100, 100]
            self.box_range = [10, 80]
            self.box_num = 20
            self.fact_type = 'tap_fake'
            self.data_type = 'rand'  # will be overridden
            self.rotate_axes = ['x', 'y', 'z']
            self.world_type = 'real'
            self.container_type = 'single'
            self.pack_type = 'last'
            self.stable_rule = 'hard_after_pack'
            self.stable_predict = 1
            self.reward_type = 'C'
            self.use_bridge = 0
            self.min_ems_width = 0
            self.min_height_diff = 0
            self.same_height_threshold = 0
            self.ems_type = 'ems-id'
            self.gripper_size = None
            self.require_box_num = 0
    
    return Args()

def generate_data(data_type, num_samples=1000, output_dir="./benchmark_data"):
    """Generate data for a specific data type"""
    
    args = get_default_args()
    args.data_type = data_type
    
    # Calculate gripper width (10% of container width as in main.py)
    gripper_width = int(np.ceil(args.container_size[0] * 0.1))
    
    # Create output directory structure
    data_dir = os.path.join(output_dir, 
                           args.fact_type, 
                           data_type, 
                           str(args.box_num),
                           f"[{args.container_size[0]}_{args.container_size[1]}]_[{args.box_range[0]}_{args.box_range[1]}]_{gripper_width}")
    
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples of {data_type} data...")
    print(f"Output directory: {data_dir}")
    print(f"Container size: {args.container_size}")
    print(f"Box range: {args.box_range}")
    print(f"Box number: {args.box_num}")
    
    # Create factory WITHOUT data_folder initially to avoid loading existing data
    factory = Factory(
        fact_type=args.fact_type,
        data_type=data_type,
        targe_container_size=args.container_size,
        gripper_width=gripper_width,
        require_box_num=args.require_box_num if args.require_box_num > 0 else None,
        data_folder=None  # Don't set data_folder initially
    )
    
    # Generate samples
    for i in tqdm.tqdm(range(num_samples)):
        # Generate new order
        factory.new_order(
            box_range=args.box_range,
            box_num=args.box_num,
            box_list=None,
            prec_graph=None
        )
        
        # Set data_folder before saving
        factory.data_folder = data_dir
        
        # Save the generated data
        factory.save_order(i)
    
    print(f"Generated {num_samples} samples for {data_type} data type")
    
    # Also save metadata
    metadata = {
        'data_type': data_type,
        'fact_type': args.fact_type,
        'container_size': args.container_size,
        'box_range': args.box_range,
        'box_num': args.box_num,
        'num_samples': num_samples,
        'gripper_width': gripper_width,
    }
    
    metadata_path = os.path.join(data_dir, "metadata.npy")
    np.save(metadata_path, metadata)
    print(f"Metadata saved to {metadata_path}")
    
    return data_dir

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark data for TAP")
    parser.add_argument("--num-samples", type=int, default=1000, 
                       help="Number of samples to generate per data type")
    parser.add_argument("--output-dir", type=str, default="./benchmark_data",
                       help="Output directory for generated data")
    parser.add_argument("--data-types", type=str, nargs="+", 
                       default=["rand", "fix", "ppsg"],
                       help="Data types to generate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TAP Benchmark Data Generation")
    print("=" * 60)
    print(f"Number of samples per data type: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data types to generate: {args.data_types}")
    print()
    
    results = {}
    for data_type in args.data_types:
        print(f"\n{'='*40}")
        print(f"Generating {data_type.upper()} data")
        print(f"{'='*40}")
        
        try:
            data_dir = generate_data(
                data_type=data_type,
                num_samples=args.num_samples,
                output_dir=args.output_dir
            )
            results[data_type] = {
                'status': 'success',
                'data_dir': data_dir,
                'num_samples': args.num_samples
            }
        except Exception as e:
            print(f"Error generating {data_type} data: {e}")
            results[data_type] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    for data_type, result in results.items():
        if result['status'] == 'success':
            print(f"{data_type.upper()}: SUCCESS - {result['num_samples']} samples in {result['data_dir']}")
        else:
            print(f"{data_type.upper()}: FAILED - {result['error']}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
