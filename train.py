from tap_train import *

# env
#     gymnasium
#     tianshou.__version__ >= 0.4.10

if __name__ == '__main__':
    args = get_args()
    
    args.box_num = 20
    args.box_range = [10, 80]
    args.container_size = [100, 100, 100]
    args.train = 1
    args.test_num = 1
    args.model = 'tnpp'
    args.prec_type = 'attn'
    args.fact_type = 'tap_fake'
    args.data_type = 'rand'
    args.ems_type = 'ems_id'
    args.stable_rule = 'hard_after_pack'
    args.rotate_axes = 'x' 'y' 'z'
    args.hidden_dim = 128
    args.world_type = 'real'
    args.container_type = 'single'
    args.pack_type = 'last'
    args.stable_predict = 1
    args.reward_type = 'C'
    args.note = 'train'
    args.device = "cuda:0"


    policy = get_policy(args)

    # print(args)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.log_path, "policy.pth"))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(args.log_path, f"checkpoint_{epoch%5}.pth")
        torch.save(policy.state_dict(), ckpt_path)
        return ckpt_path

    use_bridge = False
    test_in_train = False


    train_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(args.task, 
                box_num=args.box_num, 
                ems_dim=args.ems_dim,
                container_size=args.container_size, 
                box_range=args.box_range,
                stable_rule=args.stable_rule,
                allow_unstable=args.allow_unstable,
                use_bridge=use_bridge,
                same_height_threshold = args.same_height_threshold, 
                min_ems_width = args.min_ems_width, 
                min_height_diff = args.min_height_diff,
                fact_type=args.fact_type,
                data_type=args.data_type,
                ems_type=args.ems_type,
                rotate_axes=args.rotate_axes,
                fact_data_folder=args.fact_data_folder,
                action_type=args.action_type,
                require_box_num=args.require_box_num,
                world_type=args.world_type,
                container_type=args.container_type,
                pack_type=args.pack_type,
                ems_per_num = args.ems_per_num,
                init_ctn_num = args.init_ctn_num,
                stable_predict=args.stable_predict,
                gripper_size=args.gripper_size,
                reward_type=args.reward_type ) for _ in range(args.train_num)] )

    if args.train == 0:
        test_num = 1
    else:
        test_num = 1
    
    test_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(args.task, 
                box_num=args.box_num,  
                ems_dim=args.ems_dim,
                container_size=args.container_size, 
                box_range=args.box_range,
                stable_rule=args.stable_rule,
                allow_unstable=args.allow_unstable,
                use_bridge=use_bridge,
                same_height_threshold = args.same_height_threshold, 
                min_ems_width = args.min_ems_width, 
                min_height_diff = args.min_height_diff,
                fact_type=args.fact_type,
                data_type=args.data_type,
                ems_type=args.ems_type,
                rotate_axes=args.rotate_axes,
                action_type=args.action_type,
                require_box_num=args.require_box_num,
                world_type=args.world_type,
                container_type=args.container_type,
                pack_type=args.pack_type,
                ems_per_num = args.ems_per_num,
                init_ctn_num = args.init_ctn_num,
                stable_predict=args.stable_predict,
                gripper_size=args.gripper_size,
                reward_type='C' ) for _ in range(test_num)] )

    train_envs.seed(args.seed)
    test_envs.seed(args.seed)


    import time
    start = time.time()

    if args.train == 0:
        # Let's watch its performance!
        policy.eval()
        print(args.box_range, args.box_num)
        # test_collector = Collector(policy, test_envs)
        # test_collector.reset()
        # result = test_collector.collect(n_episode=args.test_num, render=None)
        # print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}, {len(result["rews"])}')
        run(args, test_envs, policy.actor, args.test_num)
        
    else:
        # collector
        if args.train_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(args.buffer_size)
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, test_envs)

        writer = SummaryWriter(args.log_path)
        logger = TensorboardLogger(writer)
        
        result = ts.trainer.onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch = args.max_epoch,
            step_per_epoch = args.step_per_epoch,
            repeat_per_collect = args.repeat_per_collect,
            episode_per_test = args.episode_per_test,
            batch_size = args.batch_size,
            step_per_collect = args.step_per_collect,
            test_in_train = test_in_train,
            logger = logger,
            save_best_fn = save_best_fn,
            save_checkpoint_fn = save_checkpoint_fn,
        )


        print('----over----')
        
        policy.eval()
        test_envs.seed(args.seed)
        print(args.box_range, args.box_num)
        run(args, test_envs, policy.actor, 200)

    end = time.time()
    print(args.log_path)
    print("Running time: %.2fh / %.2fm" % ((end-start) / 60.0 / 60.0, (end-start) / 60.0) )
