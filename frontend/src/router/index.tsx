import { createBrowserRouter, Navigate } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout';
import Dashboard from '../pages/Dashboard';
import Chat from '../pages/Chat';
import ModelManager from '../pages/ModelManager';
import ModelSettings from '../pages/ModelSettings';
import Orchestration from '../pages/Orchestration';
import Workflow from '../pages/Workflow';
import MultiAgent from '../pages/MultiAgent';
import Cognitive from '../pages/Cognitive';
import Training from '../pages/Training';
import VideoGen from '../pages/VideoGen';
import ImageGen from '../pages/ImageGen';
import AudioGen from '../pages/AudioGen';
import ThreeDGen from '../pages/ThreeDGen';
import PhysicsEngine from '../pages/PhysicsEngine';
import ComputerUse from '../pages/ComputerUse';
import Security from '../pages/Security';
import Federated from '../pages/Federated';
import RAG from '../pages/RAG';
import Channels from '../pages/Channels';
import Plugins from '../pages/Plugins';
import Personality from '../pages/Personality';
import DataPipeline from '../pages/DataPipeline';
import Alignment from '../pages/Alignment';
import Robot from '../pages/Robot';
import Telemetry from '../pages/Telemetry';
import Hardware from '../pages/Hardware';
import Settings from '../pages/Settings';
import Help from '../pages/Help';
import Login from '../pages/Login';
import Profile from '../pages/Profile';
import FileManager from '../pages/FileManager';
import KnowledgeBase from '../pages/KnowledgeBase';
import SkillMarket from '../pages/SkillMarket';
import PluginManager from '../pages/PluginManager';
import Notifications from '../pages/Notifications';
import ChannelConfig from '../pages/ChannelConfig';

export const router = createBrowserRouter([
  { path: '/login', element: <Login /> },
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'chat', element: <Chat /> },
      { path: 'model-manager', element: <ModelManager /> },
      { path: 'model-settings', element: <ModelSettings /> },
      { path: 'orchestration', element: <Orchestration /> },
      { path: 'workflow', element: <Workflow /> },
      { path: 'multiagent', element: <MultiAgent /> },
      { path: 'cognitive', element: <Cognitive /> },
      { path: 'training', element: <Training /> },
      { path: 'video-gen', element: <VideoGen /> },
      { path: 'image-gen', element: <ImageGen /> },
      { path: 'audio-gen', element: <AudioGen /> },
      { path: '3d-gen', element: <ThreeDGen /> },
      { path: 'physics-engine', element: <PhysicsEngine /> },
      { path: 'computer-use', element: <ComputerUse /> },
      { path: 'security', element: <Security /> },
      { path: 'federated', element: <Federated /> },
      { path: 'rag', element: <RAG /> },
      { path: 'channels', element: <Channels /> },
      { path: 'plugins', element: <Plugins /> },
      { path: 'personality', element: <Personality /> },
      { path: 'data-pipeline', element: <DataPipeline /> },
      { path: 'alignment', element: <Alignment /> },
      { path: 'robot', element: <Robot /> },
      { path: 'telemetry', element: <Telemetry /> },
      { path: 'hardware', element: <Hardware /> },
      { path: 'settings', element: <Settings /> },
      { path: 'help', element: <Help /> },
      { path: 'profile', element: <Profile /> },
      { path: 'file-manager', element: <FileManager /> },
      { path: 'knowledge-base', element: <KnowledgeBase /> },
      { path: 'skill-market', element: <SkillMarket /> },
      { path: 'plugin-manager', element: <PluginManager /> },
      { path: 'notifications', element: <Notifications /> },
      { path: 'channel-config', element: <ChannelConfig /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);
