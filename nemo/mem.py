from typing import Dict, Optional, List, Tuple, Generator
import gc
import torch
from logger import logger


import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union,Any
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import traceback
import functools
import weakref

class ModelCounter:
    """Helper class to count and analyze model parameters"""
    
    def __init__(self):
        self.param_counts = defaultdict(int)
        self.param_details = defaultdict(list)
        self.total_params = 0
        self.total_trainable = 0
        self.console = Console()
        
    def count_parameters(self, module: nn.Module, prefix: str = '') -> Tuple[int, int]:
        """
        Count parameters in a module and its submodules
        
        Args:
            module: PyTorch module to analyze
            prefix: Prefix for parameter names in detailed view
            
        Returns:
            total_params, trainable_params
        """
        total = 0
        trainable = 0
        
        # Count parameters in current module
        for name, param in module.named_parameters(prefix=prefix):
            num_params = param.numel()
            total += num_params
            
            if param.requires_grad:
                trainable += num_params
                
            # Store details
            module_name = name.split('.')[0]
            self.param_counts[module_name] += num_params
            self.param_details[module_name].append({
                'name': name,
                'shape': tuple(param.shape),
                'params': num_params,
                'trainable': param.requires_grad
            })
            
        self.total_params = total
        self.total_trainable = trainable
        return total, trainable
    
    def _format_number(self, num: int) -> str:
        """Format large numbers with commas and M/K suffixes"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num/1_000:.2f}K"
        return str(num)
    
    def print_summary(self, show_details: bool = False):
        """Print formatted parameter count summary"""
        
        # Create main summary table
        summary_table = Table(title="Model Parameter Summary")
        summary_table.add_column("Component")
        summary_table.add_column("Parameters", justify="right")
        summary_table.add_column("% of Total", justify="right")
        
        # Sort components by parameter count
        sorted_components = sorted(
            self.param_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add rows
        for component, count in sorted_components:
            percentage = (count / self.total_params) * 100
            summary_table.add_row(
                component,
                self._format_number(count),
                f"{percentage:.1f}%"
            )
            
        # Add total row
        summary_table.add_row(
            "Total",
            self._format_number(self.total_params),
            "100.0%",
            style="bold"
        )
        
        # Print summary
        self.console.print("\n")
        self.console.print(Panel(summary_table, title="Parameter Count Analysis"))
        
        # Print trainable ratio
        trainable_ratio = (self.total_trainable / self.total_params) * 100
        self.console.print(f"\nTrainable Parameters: {self._format_number(self.total_trainable)} ({trainable_ratio:.1f}%)")
        
        # Print detailed breakdown if requested
        if show_details:
            self._print_detailed_breakdown()
            
    def _print_detailed_breakdown(self):
        """Print detailed parameter breakdown by component"""
        for component, details in self.param_details.items():
            detail_table = Table(title=f"\n{component} Detailed Breakdown")
            detail_table.add_column("Layer")
            detail_table.add_column("Shape")
            detail_table.add_column("Parameters", justify="right")
            detail_table.add_column("Trainable")
            
            for param in details:
                detail_table.add_row(
                    param['name'],
                    str(param['shape']),
                    self._format_number(param['params']),
                    "✓" if param['trainable'] else "✗"
                )
                
            self.console.print(detail_table)
            
    def validate_target_params(self, target: int, tolerance: float = 0.05) -> bool:
        """
        Validate if total parameters are within tolerance of target
        
        Args:
            target: Target parameter count
            tolerance: Acceptable deviation as percentage (default 5%)
            
        Returns:
            bool: Whether parameters are within tolerance
        """
        min_params = target * (1 - tolerance)
        max_params = target * (1 + tolerance)
        
        within_range = min_params <= self.total_params <= max_params
        
        # Print validation result
        self.console.print("\nParameter Count Validation:")
        self.console.print(f"Target: {self._format_number(target)}")
        self.console.print(f"Actual: {self._format_number(self.total_params)}")
        self.console.print(f"Status: {'✓ Within tolerance' if within_range else '✗ Outside tolerance'}")
        
        return within_range
        

def clean_memory(threshold_mb: float = 1.0, clean_no_grad: bool = True):
    """
    Find large tensors that don't require gradients and clean them
    Args:
        threshold_mb: Only show tensors larger than this size in MB
        clean_no_grad: Whether to delete tensors that don't require gradients
    """
    memory_manager = TensorMemoryManager.get_instance()
    cleaned_count = 0
    cleaned_mb = 0

    # Track tensors to clean
    tensors_to_clean = []
    
    logger.info("\n=== Finding No-Grad Tensors ===")
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                # Calculate size
                tensor_size = obj.element_size() * obj.nelement()
                size_mb = tensor_size / (1024 * 1024)
                
                if size_mb < threshold_mb:
                    continue

                device = obj.device if hasattr(obj, 'device') else 'N/A'
                requires_grad = obj.requires_grad if hasattr(obj, 'requires_grad') else False
                
                if not requires_grad and id(obj) not in memory_manager.saved_tensors:
                    tensors_to_clean.append({
                        'tensor': obj,
                        'size_mb': size_mb,
                        'shape': obj.shape,
                        'device': device
                    })
                    cleaned_mb += size_mb
                    cleaned_count += 1

        except Exception as e:
            continue

    # Only log and clean if we found tensors with meaningful size
    if clean_no_grad and tensors_to_clean and cleaned_mb > 0:
        logger.info(f"\nCleaning {cleaned_count} tensors totaling {cleaned_mb:.2f}MB")
        for info in tensors_to_clean:
            try:
                # if info['size_mb'] > 1:  # Only log tensors with size > 0
                #     logger.info(f"Cleaning tensor: Shape={info['shape']}, Size={info['size_mb']:.2f}MB")
                del info['tensor']
            except:
                continue
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
def memory_stats(threshold_mb: float = 1.0):
    """
    Print all tensors in memory and their sizes
    Args:
        threshold_mb: Only show tensors larger than this size in MB
    """
    import traceback
    import sys
    import inspect
    from collections import defaultdict

    # Track total memory by type
    total_size = 0
    type_sizes = defaultdict(int)
    tensor_counts = defaultdict(int)
    
    # Get the current frame
    current_frame = sys._getframe()
    
    logger.info("\n=== Memory Analysis ===")
    logger.info(f"Showing tensors larger than {threshold_mb}MB")
    
    # Sort tensors by size for ordered output
    tensor_infos = []
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                # Calculate size
                tensor_size = obj.element_size() * obj.nelement()
                size_mb = tensor_size / (1024 * 1024)
                
                if size_mb < threshold_mb:
                    continue
                    
                # Get tensor info
                device = obj.device if hasattr(obj, 'device') else 'N/A'
                requires_grad = obj.requires_grad if hasattr(obj, 'requires_grad') else False
                
                # Try to get variable name and source
                tensor_name = 'Unknown'
                source_file = 'Unknown'
                source_line = 0
                
                # Look through referrers to find source
                for referrer in gc.get_referrers(obj):
                    if isinstance(referrer, dict):
                        # Look for the tensor in frame locals/globals
                        for name, val in referrer.items():
                            if val is obj:
                                tensor_name = name
                                # Try to get source
                                if current_frame:
                                    try:
                                        source_file = current_frame.f_code.co_filename
                                        source_line = current_frame.f_lineno
                                    except:
                                        pass
                                break
                
                # Get memory status
                if torch.cuda.is_available() and device.type == 'cuda':
                    memory_status = {
                        'allocated': torch.cuda.memory_allocated(device),
                        'cached': torch.cuda.memory_reserved(device)
                    }
                else:
                    memory_status = None
                
                tensor_infos.append({
                    'size_mb': size_mb,
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'device': device,
                    'requires_grad': requires_grad,
                    'name': tensor_name,
                    'source_file': source_file,
                    'source_line': source_line,
                    'memory_status': memory_status
                })
                
                # Update totals
                total_size += tensor_size
                type_sizes[str(obj.dtype)] += tensor_size
                tensor_counts[str(obj.dtype)] += 1
                
        except Exception as e:
            continue

    # Sort by size
    tensor_infos.sort(key=lambda x: x['size_mb'], reverse=True)
    
    # Print individual tensors
    logger.info("\n=== Individual Tensors ===")
    for info in tensor_infos:
        status_str = ''
        if info['memory_status']:
            status_str = f" | CUDA Allocated: {info['memory_status']['allocated']/1024/1024:.1f}MB, Cached: {info['memory_status']['cached']/1024/1024:.1f}MB"
            
        logger.info(
            f"Size: {info['size_mb']:.2f}MB | "
            f"Shape: {info['shape']} | "
            f"Type: {info['dtype']} | "
            f"Device: {info['device']} | "
            f"Requires Grad: {info['requires_grad']} | "
            f"Name: {info['name']} | "
            f"Source: {info['source_file']}:{info['source_line']}"
            f"{status_str}"
        )
    
    # Print summary by type
    logger.info("\n=== Memory by Type ===")
    for dtype, size in sorted(type_sizes.items(), key=lambda x: x[1], reverse=True):
        count = tensor_counts[dtype]
        logger.info(
            f"Type: {dtype} | "
            f"Total Size: {size/1024/1024:.2f}MB | "
            f"Count: {count} | "
            f"Average Size: {(size/count)/1024/1024:.2f}MB"
        )
    
    # Print total
    logger.info(f"\nTotal tensor memory: {total_size/1024/1024:.2f}MB")
    
    # Print CUDA memory summary if available
    if torch.cuda.is_available():
        logger.info("\n=== CUDA Memory Summary ===")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(i)/1024/1024:.2f}MB")
            logger.info(f"  Cached  : {torch.cuda.memory_reserved(i)/1024/1024:.2f}MB")
            logger.info(f"  Free    : {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i))/1024/1024:.2f}MB")
            
        # Add memory fragmentation info
        memory_stats = torch.cuda.memory_stats()
        if 'active_bytes.all.current' in memory_stats:
            fragmentation = (memory_stats['reserved_bytes.all.current'] - memory_stats['active_bytes.all.current']) / 1024/1024
            logger.info(f"\nMemory Fragmentation: {fragmentation:.2f}MB")

# Example usage:
# print_tensor_sizes(threshold_mb=5.0)  # Only show tensors larger than 5MB


def get_gpu_memory_map():
    """Get per-device memory usage"""
    result = {}
    try:
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)
            result[i] = {
                'allocated': allocated / 1024**2,  # MB
                'cached': cached / 1024**2,  # MB
                'free': (torch.cuda.get_device_properties(i).total_memory - allocated) / 1024**2
            }
    except Exception as e:
        logger.error(f"Error getting GPU memory: {e}")
    return result

def old_memory_stats():


    """Print detailed memory statistics"""
    memory_stats = torch.cuda.memory_stats()
    logger.info("\n=== Memory Statistics ===")
    logger.info(f"Allocated memory: {memory_stats['allocated_bytes.all.current']/1024**2:.2f}MB")
    logger.info(f"Active memory: {memory_stats['active_bytes.all.current']/1024**2:.2f}MB")
    logger.info(f"Reserved memory: {memory_stats['reserved_bytes.all.current']/1024**2:.2f}MB")
    logger.info(f"Inactive split memory: {memory_stats['inactive_split_bytes.all.current']/1024**2:.2f}MB")



class TensorMemoryManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TensorMemoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not TensorMemoryManager._initialized:
            self.saved_tensors = {}
            self.tensor_metadata = {}
            TensorMemoryManager._initialized = True
            logger.info("Initialized TensorMemoryManager singleton")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TensorMemoryManager()
        return cls._instance

    def register_tensor(self, tensor: torch.Tensor, name: str, keep_grad: bool = True):
        """Register a tensor that needs to be kept"""
        try:
            tensor_id = id(tensor)
            self.saved_tensors[tensor_id] = tensor
            self.tensor_metadata[tensor_id] = {
                'name': name,
                'shape': tensor.shape,
                'device': tensor.device,
                'requires_grad': tensor.requires_grad,
                'keep_grad': keep_grad
            }
            logger.debug(f"Registered tensor '{name}' - Shape: {tensor.shape}, "
                        f"Requires grad: {tensor.requires_grad}")
        except Exception as e:
            logger.error(f"Error registering tensor {name}: {str(e)}")


    def unregister_all(self):
        """Unregister all tracked tensors with thorough cleanup"""
        try:
            # Get list of tensor IDs before clearing
            tensor_ids = list(self.saved_tensors.keys())
            
            # Explicitly delete each tensor
            for tensor_id in tensor_ids:
                if tensor_id in self.saved_tensors:
                    # Get tensor before deleting from dict
                    tensor = self.saved_tensors[tensor_id]
                    # Delete from dictionaries
                    del self.saved_tensors[tensor_id]
                    if tensor_id in self.tensor_metadata:
                        del self.tensor_metadata[tensor_id]
                    # Explicitly delete tensor if it exists
                    if tensor is not None:
                        del tensor
            
            # Clear any remaining references
            self.saved_tensors.clear()
            self.tensor_metadata.clear()
            
            # Force immediate cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.debug(f"Unregistered all tensors with explicit cleanup")
            
        except Exception as e:
            logger.error(f"Error unregistering tensors: {str(e)}")
            logger.error(traceback.format_exc())

    def unregister_tensor(self, name: str):
        """Unregister a specific tensor by name"""
        try:
            # Find tensor id by name
            tensor_id = None
            for tid, metadata in self.tensor_metadata.items():
                if metadata['name'] == name:
                    tensor_id = tid
                    break
                    
            if tensor_id:
                # Remove tensor and metadata
                if tensor_id in self.saved_tensors:
                    del self.saved_tensors[tensor_id]
                if tensor_id in self.tensor_metadata:
                    del self.tensor_metadata[tensor_id]
                
                # Force cleanup
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.debug(f"Unregistered tensor '{name}'")
            else:
                logger.warning(f"No tensor found with name '{name}'")
                
        except Exception as e:
            logger.error(f"Error unregistering tensor '{name}': {str(e)}")
            
    def cleanup(self):
        """Clean up tensors that don't require gradients with proper reference handling"""
        try:
            initial_memory = torch.cuda.memory_allocated()
            cleaned_count = 0
            retained_count = 0
            
            # Get all objects and create a list to avoid weak reference issues
            tensor_objects = []
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor):
                        tensor_objects.append(obj)
                except ReferenceError:
                    continue
            
            # Process the collected tensors
            for tensor in tensor_objects:
                try:
                    # Keep tensors that require gradients
                    if hasattr(tensor, 'requires_grad') and not tensor.requires_grad:
                        tensor_id = id(tensor)
                        # Only delete if not explicitly saved
                        if tensor_id not in self.saved_tensors:
                            try:
                                tensor_shape = tensor.shape
                                tensor_size = tensor.element_size() * tensor.nelement() / 1024**2
                                # logger.debug(f"Cleaning tensor: Shape={tensor_shape}, Size={tensor_size:.2f}MB")
                                del tensor
                                cleaned_count += 1
                            except Exception:
                                continue
                    else:
                        retained_count += 1
                except (ReferenceError, AttributeError):
                    continue
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            freed_memory = initial_memory - final_memory
            
            logger.info(f"Memory cleaned up: {freed_memory / 1024**2:.2f}MB")
            logger.info(f"Cleaned tensors: {cleaned_count}")
            logger.info(f"Retained grad tensors: {retained_count}")
            logger.info(f"Explicitly saved tensors: {len(self.saved_tensors)}")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            logger.error(traceback.format_exc())


    def analyze_memory(self):
        """Analyze current memory usage with safe reference handling"""
        try:
            # Initialize counters
            memory_stats = {
                'gradient_tensors': {'count': 0, 'size': 0},
                'model_parameters': {'count': 0, 'size': 0},
                'other_tensors': {'count': 0, 'size': 0}
            }
            
            # Create stable list of tensors first
            tensor_list = []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        # Create strong reference
                        tensor_list.append((obj, id(obj)))
                except ReferenceError:
                    continue
                    
            # Now process the stable list
            largest_tensors = []
            for tensor, tensor_id in tensor_list:
                try:
                    # Skip if tensor was collected
                    if not tensor.is_leaf:
                        continue
                        
                    # Calculate size
                    size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                    
                    # Skip tiny tensors
                    if size_mb < 1.0:
                        continue
                        
                    tensor_info = {
                        'shape': tensor.shape,
                        'size_mb': size_mb,
                        'device': tensor.device,
                        'requires_grad': tensor.requires_grad if hasattr(tensor, 'requires_grad') else False
                    }
                    
                    # Categorize
                    if tensor_info['requires_grad']:
                        memory_stats['gradient_tensors']['count'] += 1
                        memory_stats['gradient_tensors']['size'] += size_mb
                    elif tensor_id in self.saved_tensors:
                        memory_stats['model_parameters']['count'] += 1
                        memory_stats['model_parameters']['size'] += size_mb
                    else:
                        memory_stats['other_tensors']['count'] += 1
                        memory_stats['other_tensors']['size'] += size_mb
                        
                    largest_tensors.append(tensor_info)
                    
                except (ReferenceError, AttributeError):
                    continue
                    
            # Sort and display results
            largest_tensors.sort(key=lambda x: x['size_mb'], reverse=True)
            
            # Print analysis
            logger.info("\n=== Memory Analysis ===")
            
            # CUDA Memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                logger.info(f"\nCUDA Memory:")
                logger.info(f"- Allocated: {allocated:.2f}MB")
                logger.info(f"- Reserved:  {reserved:.2f}MB")
            
            # Category breakdown
            total_size = sum(cat['size'] for cat in memory_stats.values())
            logger.info(f"\nTensor Categories:")
            for category, stats in memory_stats.items():
                if stats['count'] > 0:
                    logger.info(f"\n{category.replace('_', ' ').title()}:")
                    logger.info(f"- Count: {stats['count']}")
                    logger.info(f"- Size:  {stats['size']:.2f}MB")
                    logger.info(f"- Share: {(stats['size']/total_size*100):.1f}%")
            
            # Largest tensors
            if largest_tensors:
                logger.info("\nLargest Tensors:")
                for i, tensor in enumerate(largest_tensors[:10]):
                    if tensor['size_mb'] > 0:
                        logger.info(
                            f"{i+1}. Shape: {tensor['shape']}, "
                            f"Size: {tensor['size_mb']:.2f}MB, "
                            f"Grad: {tensor['requires_grad']}"
                        )
            
        except Exception as e:
            logger.error(f"Error in memory analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def print_memory_status(self):
        """Print current memory usage and registered tensors"""
        logger.info("\n=== Memory Status ===")
        logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        logger.info(f"CUDA Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        logger.info("\nRegistered Tensors:")
        for tensor_id, metadata in self.tensor_metadata.items():
            logger.info(f"- {metadata['name']}: "
                       f"Shape={metadata['shape']}, "
                       f"Device={metadata['device']}, "
                       f"Requires_grad={metadata['requires_grad']}")

    def clear_all(self):
        """Clear all registered tensors"""
        self.saved_tensors.clear()
        self.tensor_metadata.clear()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleared all registered tensors")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class BassTrace(nn.Module):
    """
    A mixin class that adds tensor tracking capabilities to any nn.Module.
    Usage: class YourModel(TensorTrackerMixin, nn.Module):
    """
    
    def __init__(self, *args, track_prefix: str = "", **kwargs):
        """
        Initialize the tensor tracking capabilities.
        
        Args:
            track_prefix: Prefix to add to tensor names for identification
            *args, **kwargs: Passed to parent class
        """
        # Initialize parent class first
        super().__init__(*args, **kwargs)
        
        # Initialize tracking
        self.track_prefix = track_prefix
        self.memory_manager = TensorMemoryManager.get_instance()
        self._setup_tracking()
    
    def _setup_tracking(self):
        """Set up hooks for tracking tensor operations"""
        # Store weak references to hooks to prevent memory leaks
        self._hooks = []
        
        # Set up hooks for all submodules
        for name, submodule in self.named_modules():
            # Skip the root module (self) to avoid duplicate tracking
            if submodule is self:
                continue
                
            # Register forward hook
            hook = submodule.register_forward_hook(
                functools.partial(self._forward_hook, name=name)
            )
            self._hooks.append(weakref.ref(hook))
            
            # Register parameter hooks
            self._setup_parameter_hooks(submodule, name)
            
            # Register buffer hooks if any
            self._setup_buffer_hooks(submodule, name)
    
    def _setup_parameter_hooks(self, module: nn.Module, module_name: str):
        """Set up hooks for module parameters"""
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                hook = param.register_hook(
                    functools.partial(self._gradient_hook, 
                                    name=f"{module_name}.{param_name}")
                )
                self._hooks.append(weakref.ref(hook))
    
    def _setup_buffer_hooks(self, module: nn.Module, module_name: str):
        """Set up hooks for module buffers (like batch norm running mean/var)"""
        for buffer_name, buffer in module.named_buffers(recurse=False):
            self.memory_manager.register_tensor(
                buffer,
                f"{self.track_prefix}{module_name}.{buffer_name}",
                keep_grad=False
            )
    
    def _forward_hook(self, module: nn.Module, 
                     inputs: Union[Tuple, List], 
                     output: Any, 
                     name: str):
        """Hook called after each forward pass to track output tensors"""
        def register_tensor_recursive(obj: Any, base_name: str):
            if isinstance(obj, torch.Tensor):
                self.memory_manager.register_tensor(
                    obj,
                    f"{self.track_prefix}{base_name}",
                    keep_grad=obj.requires_grad
                )
            elif isinstance(obj, (tuple, list)):
                for idx, item in enumerate(obj):
                    register_tensor_recursive(item, f"{base_name}_{idx}")
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    register_tensor_recursive(item, f"{base_name}_{key}")
        
        register_tensor_recursive(output, f"{name}_output")
    
    def _gradient_hook(self, grad: torch.Tensor, name: str):
        """Hook called when gradients are computed"""
        if grad is not None:
            self.memory_manager.register_tensor(
                grad,
                f"{self.track_prefix}{name}_grad",
                keep_grad=True
            )
        return grad
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with tensor tracking.
        Must be implemented by child class.
        """
        # Track input tensors
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                self.memory_manager.register_tensor(
                    arg,
                    f"{self.track_prefix}input_{idx}",
                    keep_grad=arg.requires_grad
                )
        
        for name, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                self.memory_manager.register_tensor(
                    arg,
                    f"{self.track_prefix}input_{name}",
                    keep_grad=arg.requires_grad
                )
        
        # Call parent's forward
        output = super().forward(*args, **kwargs)
        
        # Track output tensors
        def register_output_recursive(obj: Any, base_name: str):
            if isinstance(obj, torch.Tensor):
                self.memory_manager.register_tensor(
                    obj,
                    f"{self.track_prefix}{base_name}",
                    keep_grad=obj.requires_grad
                )
            elif isinstance(obj, (tuple, list)):
                for idx, item in enumerate(obj):
                    register_output_recursive(item, f"{base_name}_{idx}")
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    register_output_recursive(item, f"{base_name}_{key}")
        
        register_output_recursive(output, "final_output")
        
        return output
    
    def analyze(self):
        """Run memory analysis"""
        self.memory_manager.analyze_memory()
        self.memory_manager.print_memory_status()
    
    # def cleanup(self):
    #     """Clean up unused tensors and hooks"""
    #     # Remove dead hooks
    #     self._hooks = [hook for hook in self._hooks if hook() is not None]
        
    #     # Clean up tensors
    #     self.memory_manager.cleanup()

    def cleanup(self):
        """Clean up tensors with accurate memory reporting"""
        try:
            initial_memory = torch.cuda.memory_allocated()
            cleaned_count = 0
            cleaned_memory_mb = 0  # Track actual cleaned memory
            retained_count = 0
            
            # Get initial tensors
            tensor_objects = []
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor):
                        tensor_objects.append(obj)
                except ReferenceError:
                    continue
            
            # Process tensors
            for tensor in tensor_objects:
                try:
                    # Only handle CUDA tensors
                    if not tensor.is_cuda:
                        continue
                        
                    tensor_id = id(tensor)
                    size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                    
                    # Skip negligible tensors
                    if size_mb < 0.1:  # Skip very small tensors
                        continue
                    
                    # Check if we can clean this tensor
                    if (not tensor.requires_grad and 
                        tensor_id not in self.saved_tensors):
                        # Log before deletion
                        # if size_mb >= 1.0:  # Only log significant tensors
                        #     logger.info(f"Cleaning tensor: Shape={tensor.shape}, Size={size_mb:.2f}MB")
                        
                        # Track size before deletion
                        cleaned_memory_mb += size_mb
                        cleaned_count += 1
                        
                        # Delete tensor
                        del tensor
                    else:
                        retained_count += 1
                        
                except (ReferenceError, AttributeError, RuntimeError):
                    continue
                    
            # Force cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            # Calculate actual freed memory
            final_memory = torch.cuda.memory_allocated()
            actual_freed_mb = (initial_memory - final_memory) / (1024 * 1024)
            
            # Report cleanup results
            logger.info("\nMemory Cleanup Summary:")
            logger.info(f"- Tensors cleaned: {cleaned_count}")
            logger.info(f"- Total size of cleaned tensors: {cleaned_memory_mb:.2f}MB")
            logger.info(f"- Actually freed memory: {actual_freed_mb:.2f}MB")
            logger.info(f"- Retained tensors: {retained_count}")
            logger.info(f"- Saved tensors: {len(self.saved_tensors)}")
            
            # Report CUDA memory status
            if torch.cuda.is_available():
                logger.info("\nCUDA Memory Status:")
                logger.info(f"- Currently allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                logger.info(f"- Maximum allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f}MB")
                logger.info(f"- Currently reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
                
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            
            
    def __del__(self):
        """Clean up hooks when the module is deleted"""
        for hook_ref in self._hooks:
            hook = hook_ref()
            if hook is not None:
                hook.remove()
        self._hooks.clear()