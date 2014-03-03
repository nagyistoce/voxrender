/* ===========================================================================

	Project: VoxRender - Main Window Interface

	Description: Implements the main window for the VoxRender GUI

    Copyright (C) 2012-2014 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Include Headers
#include "mainwindow.h"
#include "ui_mainwindow.h"

// Include Dependencies
#include "aboutdialogue.h"
#include "ambientlightwidget.h"
#include "arealightwidget.h"
#include "camerawidget.h"
#include "infowidget.h"
#include "pointlightwidget.h"
#include "lightdialogue.h"
#include "clipdialogue.h"
#include "clipplanewidget.h"
#include "histogramgenerator.h"
#include "animatewidget.h"
#include "pluginwidget.h"
#include "timingwidget.h"

// VoxLib Includes 
#include "VoxLib/IO/ResourceHelper.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/Core/System.h"
#include "VoxLib/Core/Logging.h"

// VoxScene Bundle
#include "VoxScene/Camera.h"
#include "VoxScene/Light.h"
#include "VoxScene/Animator.h"
#include "VoxScene/PrimGroup.h"
#include "VoxScene/RenderParams.h"
#include "VoxScene/Scene.h"
#include "VoxScene/Transfer.h"
#include "VoxScene/Volume.h"
#include "VoxScene/FrameBuffer.h"
#include "VoxScene/Renderer.h"
#include "VoxScene/RenderThread.h"

// Qt Includes
#include <QtCore/QDateTime>
#include <QtCore/QTextStream>
#include <QtGui/QStandardItemModel>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <QtGui/QTextLayout>
#include <QtGui/QClipboard>
#include <QSettings>

// Volume Transform Includes
#include "VoxVolt/Conv.h"

#define VOX_GUI_LOG_CAT "GUI"

// Singleton pointer
MainWindow* MainWindow::instance;

using namespace vox;

// Filescope namespace
namespace {
namespace filescope 
{
    const int logTabId = 3;  // Log tab index

    // --------------------------------------------------------------------
    //  Gets a datetime string using QDateTime
    // --------------------------------------------------------------------
    std::string datetime(QString format)
    {
        QString result = QDateTime::currentDateTime().toString(format);
        return std::string( result.toUtf8().data() );
    }

    // --------------------------------------------------------------------
    //  Performs ellision on the path component of the input filepath
    // --------------------------------------------------------------------
    QString pathElidedText(const QFontMetrics &fm, 
                            const QString &text, 
                            int width, 
                            int flags 
                            ) 
    {
	    const QString filename = "/" + QFileInfo( text ).fileName( );
	    const QString path = QFileInfo( text ).absolutePath( );

	    int fwidth = fm.width( filename );

	    if( fwidth > width ) return fm.elidedText( text, Qt::ElideMiddle, width, flags );
            
	    return fm.elidedText( path, Qt::ElideMiddle, width - fwidth, flags ) + filename;
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
//  Setup the logging backend and instantiate the environment
// ----------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this); instance = this;

    // Register the QT metatypes used as signals parameters
    QMetaTypeId<std::shared_ptr<vox::FrameBufferLock>>::qt_metatype_id();
    QMetaTypeId<std::string>::qt_metatype_id();
    
    // VoxRender log configuration
    configureLoggingEnvironment();

    // Display and log the library version info and startup time
    VOX_LOG_INFO(VOX_GUI_LOG_CAT, format("VoxRender Version: %1%", VOX_VERSION_STRING));
    
    // Load the configuration file from the current directory
    try
    {
        vox::ResourceHelper::loadConfigFile("VoxRender.config");
    }
    catch (Error & error)
    {
        VOX_LOG_ERROR(error.code, VOX_GUI_LOG_CAT, "Unable to load config file: " + error.message);
    }

	// Window Initialization
	createRenderTabPanes();
	createDeviceTable();
    createRecentFileActions();
    createStatusBar();
    configurePlugins();

	// Create transfer function widget
	transferwidget = new TransferWidget();
	ui->transferAreaLayout->addWidget(transferwidget);

	// Configure the render view + interaction widget
	m_renderView = new RenderView(ui->frame_render);
	ui->frame_render_layout->addWidget(m_renderView, 0, 0, 1, 1);
    connect(this, SIGNAL(frameReady(std::shared_ptr<vox::FrameBufferLock>)),
            m_infowidget, SLOT(updatePerfStats(std::shared_ptr<vox::FrameBufferLock>)));
    connect(this, SIGNAL(frameReady(std::shared_ptr<vox::FrameBufferLock>)), 
            m_renderView, SLOT(setImage(std::shared_ptr<vox::FrameBufferLock>)));
    connect(m_renderView, SIGNAL(viewChanged(float)), this, SLOT(onZoomChange(float)));
    onZoomChange(m_renderView->zoomFactor()); // Ensure the initial zoom display is correct
    
    connect(this, SIGNAL(progressChanged(int)), this, SLOT(onProgressChanged(int)));

    readSettings(); // Read in the application settings

    // Initialize the master renderer and register the interactive display callback
    m_renderer = vox::VolumeScatterRenderer::create();
    m_renderer->setRenderEventCallback(std::bind(&MainWindow::onFrameReady, 
        this, std::placeholders::_1));

	// Set initial render state 
    changeRenderState(RenderState_Waiting);
}

// ----------------------------------------------------------------------------
//  Write the application settings file and terminate the core library
// ----------------------------------------------------------------------------
MainWindow::~MainWindow()
{
    writeSettings();

    HistogramGenerator::instance()->stopGeneratingImages();

    m_activeScene.reset();
    m_renderController.stop();
    m_renderer.reset();

    delete histogramwidget;
    delete transferwidget;
	delete m_renderView;

    vox::PluginManager::instance().unloadAll();

    delete ui;
}

// ----------------------------------------------------------------------------
//  Configures the logging system, called immediately on startup
// ----------------------------------------------------------------------------
void MainWindow::configureLoggingEnvironment()
{
	// Register the GUI logger backend with the frontend
    vox::Logger::setHandler(std::bind(&MainWindow::voxGuiErrorHandler, this, 
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, 
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));

    // Create the log signal connections before any logging calls are made
    connect(&m_logBlinkTimer, SIGNAL(timeout()), this, SLOT(blinkTrigger()));
    connect(this, SIGNAL(logEntrySignal(char const*, int, int, int, char const*, std::string)), 
            this, SLOT(printLogEntry(char const*, int, int, int, char const*, std::string)));

    // Compose default filename and path for session log file
    String logLocation = vox::System::currentDirectory() + "/Logs/"; 
    String logFilename = vox::System::computerName() + "_" +
                         filescope::datetime("yyyy-MM-dd_hh.mm.ss") + 
                         ".log";
    
    // Ensure log directory exists for local filesystem 
    if (!boost::filesystem::exists(logLocation))
    {
        boost::filesystem::create_directory(logLocation);
    }

    // Create log file for this session 
    m_logFileStream.open(logLocation + logFilename, std::ios_base::app);

    // Redirect std::clog stream to session log sink
    if (m_logFileStream)
    {
        std::clog.set_rdbuf(m_logFileStream.rdbuf());
    }
    else
    {
        VOX_LOG_WARNING(Error_System, VOX_GUI_LOG_CAT,
            "Unable to establish output stream to log file");
    }
}

// ----------------------------------------------------------------------------
//  Enumerates the available plugins in the plugin window for user selection
// ----------------------------------------------------------------------------
void MainWindow::configurePlugins()
{
    m_pluginSpacer = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
	ui->pluginsAreaLayout->addItem(m_pluginSpacer);

    // Enumerate all available plugins and add them to the display panel
    vox::PluginManager::instance().forEach(boost::bind(&MainWindow::registerPlugin, this, _1));
}

// ----------------------------------------------------------------------------
//  Adds a new plugin pane to the list of available plugins
// ----------------------------------------------------------------------------
void MainWindow::registerPlugin(std::shared_ptr<PluginInfo> info)
{    
    // Remove spacer prior to pane insertion
    ui->pluginsAreaLayout->removeItem(m_pluginSpacer);

    // Create new pane for the plugin widget
    PaneWidget *pane = new PaneWidget(ui->pluginsAreaContents);
    pane->showOnOffButton();
    pane->setTitle(QString::fromLatin1(info->name.c_str()));

    // Create the plugin widget
    QWidget * newPluginWidget = new PluginWidget(pane, info); 
    pane->setWidget(newPluginWidget);

    ui->pluginsAreaLayout->addWidget(pane);

    m_pluginPanes.push_back(pane);

    // Reinsert spacer following new pane
	ui->pluginsAreaLayout->addItem(m_pluginSpacer);
}

// ----------------------------------------------------------------------------
//  Outputs log entries to the log TextEdit and the logger backend
// ----------------------------------------------------------------------------
void MainWindow::voxGuiErrorHandler(
    char const* file, int line, int severity, int code, 
    char const* category, char const* message)
{
    // Output the log entry to the registered backend (std::clog)
	vox::ErrorPrint(file, line, severity, code, category, message);

    // Signal the need to write an additional entry to the log page
    emit logEntrySignal(file, line, severity, code, category, message);
}

// ----------------------------------------------------------------------------
//  Outputs log entries to the log TextEdit 
// ----------------------------------------------------------------------------
void MainWindow::printLogEntry(
    char const* file, int line, int severity, int code, 
    char const* category, std::string message)
{
	static const QColor debugColour   = Qt::black;
	static const QColor infoColour    = Qt::blue;
	static const QColor warningColour = Qt::darkMagenta;
	static const QColor errorColour   = Qt::red;
	static const QColor severeColour  = Qt::red;

    // Reposition the write head to the end of the log
	QTextCursor cursor = ui->textEdit_log->textCursor();
	cursor.movePosition(QTextCursor::End);

    // Timestamp the log entry
	QTextStream ss(new QString());
	ss << '[' << QDateTime::currentDateTime().toString(tr("yyyy-MM-dd hh:mm:ss")) << ' ';
	bool warning = false; bool error = false;

	// Append log message to end of document
	QColor textColor;
	switch(severity) {
		case vox::Severity_Debug:
			ss << tr("Debug: ");
			textColor = debugColour;
			break;
		case vox::Severity_Info:
		default:
			ss << tr("Info: ");
			textColor = infoColour;
			break;
		case vox::Severity_Warning:
			ss << tr("Warning: ");
			textColor = warningColour;
			warning = true;
			break;
		case vox::Severity_Error:
			ss << tr("Error: ");
			textColor = errorColour;
			error = true;
			break;
		case vox::Severity_Fatal:
			ss << tr("Severe Error: ");
			textColor = severeColour;
			break; }

	ss << code << "] ";
	ss.flush();

	// Insert formatted error information
	QTextCharFormat fmt(cursor.charFormat());
	fmt.setForeground(QBrush(textColor));
	cursor.setCharFormat(fmt);
	cursor.insertText(ss.readAll());

	// Insert error message string
	fmt.setForeground(QBrush(Qt::black));
	cursor.setCharFormat(fmt);
    ss << QString(message.c_str()) << endl;
	cursor.insertText(ss.readAll());

	// Activate flip-flop for logged error notification icon
    int currentIndex = ui->tabWidget_main->currentIndex();
	if (currentIndex != filescope::logTabId && severity > vox::Severity_Info)
    {
		m_logBlinkState = true;

		if (severity < vox::Severity_Error) 
        {
			static const QIcon icon(":/icons/warningicon.png");
            ui->tabWidget_main->setTabIcon(filescope::logTabId, icon);
		} 
        else 
        {
			blinkTrigger();
			static const QIcon icon(":/icons/erroricon.png");
            ui->tabWidget_main->setTabIcon(filescope::logTabId, icon);
        } 
    }
}

// ----------------------------------------------------------------------------
//  Reads the program settings file through the Qt4 library
// ----------------------------------------------------------------------------
void MainWindow::readSettings()
{
	QSettings settings( "VoxRender", "VoxRender GUI");

    // MainWindow settings group
	settings.beginGroup("MainWindow");
        BOOST_FOREACH (auto & file, settings.value("recentFiles").toStringList())
        {
            m_recentFiles.append( QFileInfo(file) );
        }
	    m_lastOpenDir = settings.value("lastOpenDir","").toString( );
	    updateRecentFileActions( );
	settings.endGroup( );
}

// ----------------------------------------------------------------------------
// Writes the program settings file through the Qt4 library
// ----------------------------------------------------------------------------
void MainWindow::writeSettings()
{
	QSettings settings("VoxRender", "VoxRender GUI");

	settings.beginGroup("MainWindow");
	{
		QListIterator<QFileInfo> i(m_recentFiles);
		QStringList recentFilesList;
		while( i.hasNext( ) ) 
            recentFilesList.append( i.next( ).absoluteFilePath( ) );
		settings.setValue("recentFiles", recentFilesList);
	}
	settings.setValue("lastOpenDir", m_lastOpenDir);
	settings.endGroup( );
}

// ----------------------------------------------------------------------------
// Creates the open recent file action list
// ----------------------------------------------------------------------------
void MainWindow::createRecentFileActions()
{
    // Create actions for recent files listing
	for (int i = 0; i < MaxRecentFiles; i++) 
    {
		m_recentFileActions[i] = new QAction( this );
		m_recentFileActions[i]->setVisible( false );

        // Connect action slot to open recent file to open slot
		connect(m_recentFileActions[i], SIGNAL(triggered()), 
                 this, SLOT(onActionOpenRecentFile()));
	}

    // Add the actions to the recent file submenu
	for (int i = 0; i < MaxRecentFiles; i++)
		ui->menuOpen_Recent->addAction(m_recentFileActions[i]);
}

// ----------------------------------------------------------------------------
// Updates the recent file actions listing
// ----------------------------------------------------------------------------
void MainWindow::updateRecentFileActions()
{
    // Refresh file listing and detect missing files
	QMutableListIterator<QFileInfo> i(m_recentFiles);
	while (i.hasNext()) 
    {
		i.peekNext().refresh();
		if (!i.next().exists())
			i.remove();
	}

    // Update file info listings for recent files
	for( int j = 0; j < MaxRecentFiles; j++ ) 
    {
		if( j < m_recentFiles.count( ) ) 
        {
			QFontMetrics fm( m_recentFileActions[j]->font( ) );
			QString filename = m_recentFiles[j].absoluteFilePath( );

            QString ellidedText = filescope::pathElidedText( fm, filename, 250, 0 );

			m_recentFileActions[j]->setText( ellidedText );
			m_recentFileActions[j]->setData( filename );
			m_recentFileActions[j]->setVisible( true );
		} 
        else m_recentFileActions[j]->setVisible( false );
	}
}

// ----------------------------------------------------------------------------
// Creates the application window status bar
// ----------------------------------------------------------------------------
void MainWindow::createStatusBar()
{
	activityLabel   = new QLabel(tr("  Status:"));
	activityMessage = new QLabel();
	statusLabel     = new QLabel(tr(" Activity:"));
	statusMessage   = new QLabel();
	statusProgress  = new QProgressBar();
	statsLabel      = new QLabel(tr(" Statistics:"));
	statsMessage    = new QLabel();

	activityLabel->setMaximumWidth( 60 );
	activityMessage->setFrameStyle( QFrame::Panel | QFrame::Sunken );
	activityMessage->setMaximumWidth( 140 );
	statusLabel->setMaximumWidth( 60 );
	statusMessage->setFrameStyle( QFrame::Panel | QFrame::Sunken );
	statusMessage->setMaximumWidth( 320 );
	statusProgress->setMaximumWidth( 100 );
	statusProgress->setRange( 0, 100 );
	statsLabel->setMaximumWidth( 70 );
	statsMessage->setFrameStyle( QFrame::Panel | QFrame::Sunken );

	ui->statusBar->addPermanentWidget( activityLabel, 1 );
	ui->statusBar->addPermanentWidget( activityMessage, 1 );

	ui->statusBar->addPermanentWidget( statusLabel, 1 );
	ui->statusBar->addPermanentWidget( statusMessage, 1 );
	ui->statusBar->addPermanentWidget( statusProgress, 1 );
	ui->statusBar->addPermanentWidget( statsLabel, 1 );
	ui->statusBar->addPermanentWidget( statsMessage, 1 );
}

// ----------------------------------------------------------------------------
// Updates the window with the current filename information 
// ----------------------------------------------------------------------------
void MainWindow::setCurrentFile(QString const& path)
{
    QString showName;

    // Check for path specification
	if (!path.isEmpty()) 
    {
        // Check for piped file
		if (path == "-") 
        {
            showName = "VoxRender - Piped Scene";
        }
		else 
        {
            QFileInfo info(path);

			showName = "VoxRender - " + info.fileName();
            
            // Set the most recent base directory
			m_lastOpenDir = info.absolutePath();

            // Update the recent file listings
			if (path.endsWith(".xml")) 
            {
				m_recentFiles.removeAll(info);
				m_recentFiles.prepend(info);
				updateRecentFileActions();
			}
		}
	}
    else
    {
        showName = "VoxRender - Untitled";
    }
    
	setWindowModified(false); // Disregard previous changes

    // Update the window title with the basename
	setWindowTitle(tr("%1[*]").arg(showName));
}

// ----------------------------------------------------------------------------
// Loads the specified scene file for rendering
// ----------------------------------------------------------------------------
void MainWindow::renderNewSceneFile(QString const& filename) 
{
    // Get the base filename for logging purposes
    std::string const file = boost::filesystem::path(
        filename.toUtf8().data()).filename().string();

    // New file loading info message
    VOX_LOG_INFO(VOX_GUI_LOG_CAT, format("Loading Scene File: %1%", file));

    // Update the status bar
	changeRenderState(RenderState_Loading);
    
    // Compose the resource identifier for filesystem access
    std::string identifier(filename.toUtf8().data());
    if (identifier.front() != '/') identifier = '/' + identifier;

    // Load specified scene file
    try
    {
        // Attempt to parse the scene file content
        m_activeScene = vox::Scene::imprt(identifier);
        if (!m_activeScene.volume) throw Error(__FILE__, __LINE__, VOX_GUI_LOG_CAT, "Scene is missing volume data", Error_MissingData);
        if (!m_activeScene.parameters)   m_activeScene.parameters   = RenderParams::create();
        if (!m_activeScene.clipGeometry) m_activeScene.clipGeometry = PrimGroup::create();
        if (!m_activeScene.lightSet)     m_activeScene.lightSet     = LightSet::create();
        if (!m_activeScene.transferMap)  m_activeScene.transferMap  = TransferMap::create(); // :TODO: Only required because of a bug
        if (!m_activeScene.transfer)     m_activeScene.transfer     = Transfer1D::create();
        if (!m_activeScene.camera)       m_activeScene.camera       = Camera::create();
        if (!m_activeScene.animator)     m_activeScene.animator     = Animator::create();

        // Synchronize the scene view
        synchronizeView();

        setCurrentFile(filename); // Update window name

        m_renderController.stop();
    }
    catch (vox::Error & error)
    {
        VOX_LOG_ERROR(error.code, VOX_GUI_LOG_CAT, format(
            "Failed to load %1% [%2%]", file, error.message)); 
    }
    catch (std::exception & error)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_GUI_LOG_CAT, format(
            "Unexpected exception loading %1%: %2%", file, error.what()));
    }

    beginRender();
}

// ----------------------------------------------------------------------------
//  Begins rendering the currently active scene file
// ----------------------------------------------------------------------------
void MainWindow::beginRender(size_t samples, bool animation)
{
    // Check if we are already in a rendering state
    if (m_renderController.isActive()) 
    {
        VOX_LOG_ERROR(Error_NotAllowed, VOX_GUI_LOG_CAT, "A render operation is already in progress");
        return;
    }

    // attempt to begin the render
    try
    {
        m_renderController.setProgressCallback([this] (float p) 
            { 
                int value = (int)(p*100.f);
                if (value != statusProgress->value())
                    progressChanged(value);
            });

        // Update the render state
        changeRenderState(animation ? RenderState_Animating : RenderState_Rendering);

        // Initiate rendering of the scene
        if (animation) m_renderController.render(m_renderer, m_activeScene.animator, samples);
        else
        {
            emit sceneChanged(); // Send the scene change signal

            m_renderController.render(m_renderer, m_activeScene, samples);
        }
    }
    catch (Error & error) { VOX_LOG_EXCEPTION(Severity_Error, error); }
}

// ----------------------------------------------------------------------------
//  Updates the progress bar on the main window
// ----------------------------------------------------------------------------
void MainWindow::onProgressChanged(int progress)
{
    statusProgress->setValue(progress);
}

// ----------------------------------------------------------------------------
// Sets the Vox GUI render state
// ----------------------------------------------------------------------------
void MainWindow::changeRenderState(RenderState state)
{
	m_guiRenderState = state;

	switch (state) 
	{
		case RenderState_Waiting:
			// Waiting for input file. Most controls disabled.
			ui->pushButton_clipboard->setEnabled( false );
			ui->pushButton_resume->setEnabled( false );
			ui->pushButton_pause->setEnabled( false );
			ui->pushButton_stop->setEnabled( false );
			ui->tabWidget_render->setEnabled( false );
			ui->label_zoomIcon->setVisible( false );
			ui->label_zoom->setVisible( false );
			activityMessage->setText("Idle");
			statusProgress->setRange(0,100);
			break;
        case RenderState_Loading:
			// Waiting for input file. Most controls disabled.
			ui->pushButton_clipboard->setEnabled( false );
			ui->pushButton_resume->setEnabled( false );
			ui->pushButton_pause->setEnabled( false );
			ui->pushButton_stop->setEnabled( false );
			ui->tabWidget_render->setEnabled( false );
			ui->label_zoomIcon->setVisible( false );
			ui->label_zoom->setVisible( false );
			activityMessage->setText("Parsing scene file");
			break;
		case RenderState_Rendering:
			// Rendering is in progress.
			ui->pushButton_clipboard->setEnabled( true );
			ui->pushButton_resume->setEnabled( false );
			ui->pushButton_pause->setEnabled( true );
			ui->pushButton_stop->setEnabled( true );
			ui->tabWidget_render->setEnabled( true );
			ui->label_zoomIcon->setVisible( true );
			ui->label_zoom->setVisible( true );
			activityMessage->setText("Rendering (Interactive)");
			break;
		case RenderState_Stopped:
			ui->pushButton_clipboard->setEnabled( true );
			ui->pushButton_resume->setEnabled( true );
			ui->pushButton_pause->setEnabled( false );
			ui->pushButton_stop->setEnabled( false );
			activityMessage->setText(tr("Render stopped"));
			break;
		case RenderState_Paused:
			ui->pushButton_resume->setEnabled( true );
			ui->pushButton_pause->setEnabled( false );
			ui->pushButton_stop->setEnabled( true );
			activityMessage->setText(tr("Rendering is paused"));
			break;
	}
}

// ----------------------------------------------------------------------------
// Returns true if the user allows termination
// ----------------------------------------------------------------------------
bool MainWindow::canStopRendering()
{
	if (m_guiRenderState == RenderState_Rendering) 
	{
		QMessageBox msgBox( this );
		msgBox.setIcon( QMessageBox::Question );
		msgBox.setText( tr("Do you want to stop the current render and load a new scene?") );
		msgBox.setWindowTitle( "Rendering in progress" );
	
		QPushButton* accept = msgBox.addButton( tr("Yes"), QMessageBox::AcceptRole );
		msgBox.addButton( tr("No"), QMessageBox::RejectRole );
		QPushButton* cancel = msgBox.addButton( tr("Cancel"), QMessageBox::RejectRole );
		msgBox.setDefaultButton( cancel );
		
		msgBox.exec( );

		if (msgBox.clickedButton( ) != accept) return false;
	}

	return true;
}

// ----------------------------------------------------------------------------
//  Synchronizes the current scene 
// ----------------------------------------------------------------------------
void MainWindow::synchronizeView()
{
    transferwidget->synchronizeView();

    // Synchronize the lighting controls
    BOOST_FOREACH (auto & pane, m_lightPanes) delete pane;
    m_lightPanes.clear();
    BOOST_FOREACH (auto & light, m_activeScene.lightSet->lights()) 
        addLight(light, "Point Light");

    // Synchronize the clip geometry controls
    BOOST_FOREACH (auto & pane, m_clipPanes) delete pane;
    m_clipPanes.clear();
    BOOST_FOREACH (auto & clip, m_activeScene.clipGeometry->children())
        addClippingGeometry(clip);
}

// ----------------------------------------------------------------------------
//  Creates the render tab panes
// ----------------------------------------------------------------------------
void MainWindow::createRenderTabPanes()
{
	//
	// Main Tab
	//

	// Create imaging tab panes
	panes[0] = new PaneWidget( ui->panesAreaContents, 
		"Sampling Parameters", ":/icons/samplingicon.png" );
	panes[1] = new PaneWidget( ui->panesAreaContents, 
		"Camera Settings", ":/icons/cameraicon.png" );
	panes[2] = new PaneWidget( ui->panesAreaContents, 
		"Volume Display", ":/icons/clockicon.png" );
	panes[3] = new PaneWidget( ui->panesAreaContents, 
		"Volume Histogram", ":/icons/histogramicon.png" );
    panes[4] = new PaneWidget( ui->panesAreaContents,
        "Animation Controls", ":/icons/cameraicon.png" );

	// Sampler settings widget
	samplingwidget = new SamplingWidget( panes[0] );
	panes[0]->setWidget( samplingwidget );
	ui->panesAreaLayout->addWidget( panes[0] );

	// Camera / film settings widget
	camerawidget = new CameraWidget( panes[1] );
	panes[1]->setWidget( camerawidget );
	ui->panesAreaLayout->addWidget( panes[1] );

	// Time settings widget
	timingwidget = new TimingWidget( panes[2] );
	panes[2]->setWidget( timingwidget );
	ui->panesAreaLayout->addWidget( panes[2] );

	// Volume data histogram widget
	histogramwidget = new HistogramWidget( panes[3] );
	panes[3]->setWidget( histogramwidget );
	ui->panesAreaLayout->addWidget( panes[3] );

    // Animation sequencing widget
    animationwidget = new AnimateWidget( panes[4] );
    panes[4]->setWidget(animationwidget);
    ui->panesAreaLayout->addWidget( panes[4] );

	// Set alignment of panes within main tab layout 
	ui->panesAreaLayout->setAlignment( Qt::AlignTop );
	ui->panesAreaLayout->addItem( new QSpacerItem( 20, 20, 
		QSizePolicy::Minimum, QSizePolicy::Expanding) );

	// 
	// Lighting Tab
	// 

	// Set alignment of panes within lighting tab layout 
	ui->lightsAreaLayout->setAlignment( Qt::AlignTop );
    m_spacer = new QSpacerItem( 20, 20, 
        QSizePolicy::Minimum, QSizePolicy::Expanding );

    // Create new pane for the ambient light setting widget
    m_ambientPane = new PaneWidget(ui->lightsAreaContents);
    QWidget * currWidget = new AmbientLightWidget(m_ambientPane); 

    m_ambientPane->setTitle("Environment");
    m_ambientPane->setIcon(":/icons/lightgroupsicon.png");
    m_ambientPane->setWidget(currWidget);
    m_ambientPane->expand();

    ui->lightsAreaLayout->addWidget(m_ambientPane);

    // Reinsert spacer following new pane
	ui->lightsAreaLayout->addItem( m_spacer );

    //
    // Clipping Geometry
    //

    m_clipSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);
	ui->clipAreaLayout->addItem(m_clipSpacer);

	// 
	// Advanced Tab
	//

	// Create advanced tab panes
	advpanes[0] = new PaneWidget( ui->advancedAreaContents, 
		"Scene Information", ":/icons/logtabicon.png" );

	// Scene information widget
	m_infowidget = new InfoWidget( advpanes[0] );
	advpanes[0]->setWidget( m_infowidget );
	ui->advancedAreaLayout->addWidget( advpanes[0] );

	// Set alignment of panes within advanced tab layout 
	ui->advancedAreaLayout->setAlignment( Qt::AlignTop );
	ui->advancedAreaLayout->addItem( new QSpacerItem( 20, 20, 
		QSizePolicy::Minimum, QSizePolicy::Expanding) );
}

// ----------------------------------------------------------------------------
//  Creates the CUDA device table listings
// ----------------------------------------------------------------------------
void MainWindow::createDeviceTable()
{
	// CUDA enabled device info structures
	vox::DeviceManager::loadDeviceInfo( );

	// Create headers for CUDA device properties
	static const int NUM_COLS = 10; QStringList headers;
	ui->table_devices->setColumnCount( NUM_COLS );
	headers.append( tr("Active?") );
	headers.append( tr("Device ID") );
	headers.append( tr("Name") );
	headers.append( tr("Clock Speed (MHz)") );
	headers.append( tr("Global Memory (MB)") );
	headers.append( tr("Constant Memory (KB)") );
	headers.append( tr("Shared Memory (KB/SM)") );
	headers.append( tr("L2 Cache Size (KB)") );
	headers.append( tr("Register (/SM)") );
	headers.append( tr("Time Limit?") );
	ui->table_devices->setHorizontalHeaderLabels( headers );

	// Add device listings to the device table
	int ndevices = vox::DeviceManager::getDeviceCount( );
	ui->table_devices->setRowCount( ndevices );
	for( int dev = 0; dev < ndevices; dev++ )
	{
		QTableWidgetItem* tableWidget[NUM_COLS];

		vox::Device const& device = vox::DeviceManager::getDevice(dev);

        // :TODO: printf style formatting operator
#define STRING(s) static_cast<std::ostringstream&> \
	( std::ostringstream( ) << s ).str( ).c_str( )
		std::string timeLimit = device.props.kernelExecTimeoutEnabled ? "Yes" : "No";
		tableWidget[0] = new QTableWidgetItem(tr("NO"));
		tableWidget[1] = new QTableWidgetItem(STRING(dev));
		tableWidget[2] = new QTableWidgetItem(STRING(std::string(device.props.name)));
		tableWidget[3] = new QTableWidgetItem(STRING(device.props.clockRate/1000));
		tableWidget[4] = new QTableWidgetItem(STRING(device.props.totalGlobalMem/1048576));
		tableWidget[5] = new QTableWidgetItem(STRING(device.props.totalConstMem/1024));
		tableWidget[6] = new QTableWidgetItem(STRING(device.props.sharedMemPerBlock/1024));
		tableWidget[7] = new QTableWidgetItem(STRING(device.props.l2CacheSize/1024));
		tableWidget[8] = new QTableWidgetItem(STRING(device.props.regsPerBlock));
		tableWidget[9] = new QTableWidgetItem(STRING(timeLimit));
#undef STRING

		// Format enabled indicator cell widget
		tableWidget[0]->font( ).setWeight( QFont::Black );
		tableWidget[0]->setBackgroundColor( Qt::darkRed );
		tableWidget[0]->setTextColor( Qt::white );

		for( int i = 0; i < NUM_COLS; i++ ) 
		{
			// Set center text alignment for cells
			tableWidget[i]->setTextAlignment( Qt::AlignVCenter|Qt::AlignHCenter );

			// Make the cells read-only 
			tableWidget[i]->setFlags( tableWidget[i]->flags( )^Qt::ItemIsEditable );

			// Add the cell widgets to the table
			ui->table_devices->setItem( dev, i, tableWidget[i] );
		}
	}

	// Update device count displays
    int deviceCount = 0;
	ui->label_deviceCount_1->setText( QString("%1").arg(deviceCount) );
	ui->label_deviceCount_2->setText( QString("%1").arg(deviceCount) );
}

// ----------------------------------------------------------------------------
//  Adds a control widget for an existing light object 
// ----------------------------------------------------------------------------
void MainWindow::addLight(std::shared_ptr<Light> light, QString const& name)
{
    // Remove spacer prior to pane insertion
    ui->lightsAreaLayout->removeItem(m_spacer);

    // Create new pane for the light setting widget
    PaneWidget *pane = new PaneWidget(ui->lightsAreaContents);
   
    QWidget * currWidget = new PointLightWidget(pane, light); 

    int index = m_lightPanes.size( );

    pane->SetIndex(index);
    pane->showOnOffButton();
    pane->showVisibilityButtons();
    pane->setTitle(name);
    pane->setIcon(":/icons/lightgroupsicon.png");
    pane->setWidget(currWidget);
    pane->expand( );

    connect(pane, SIGNAL(removed(PaneWidget *)), this, SLOT(removeLight(PaneWidget *)));

    ui->lightsAreaLayout->addWidget(pane);

    m_lightPanes.push_back(pane);

    // Reinsert spacer following new pane
	ui->lightsAreaLayout->addItem( m_spacer );
}

// ----------------------------------------------------------------------------
//  Removes a light from the active scene
// ----------------------------------------------------------------------------
void MainWindow::removeLight(PaneWidget * pane)
{
    m_lightPanes.remove(pane);

    ui->lightsAreaLayout->removeWidget(pane);
    delete pane;
}

// ----------------------------------------------------------------------------
//  Adds a control widget for an existing clipping geometry object 
// ----------------------------------------------------------------------------
void MainWindow::addClippingGeometry(std::shared_ptr<vox::Primitive> prim)
{
    // Remove spacer prior to pane insertion
    ui->clipAreaLayout->removeItem(m_clipSpacer);

    // Create new pane for the light setting widget
    PaneWidget *pane = new PaneWidget(ui->clipAreaContents);
   
    // Create the control widget to populate the pane
    QWidget * currWidget = nullptr;
    if (prim->typeId() == Plane::classTypeId())
    {
        auto plane = std::dynamic_pointer_cast<vox::Plane>(prim);
        if (!plane) throw Error(__FILE__, __LINE__, VOX_GUI_LOG_CAT, 
            "Error interpreting primitive :TODO:");
        currWidget = new ClipPlaneWidget(pane, plane); 
    }
    else
    {
        // :TODO: unexpandeable hideable attribute pane

        VOX_LOG_WARNING(Error_NotImplemented, VOX_GUI_LOG_CAT, 
            format("Geometry type '%1%' unrecognized. '%2%' will not be editable.", prim->typeId(), prim->idString()));

        return;
    }

    int index = m_clipPanes.size();

    pane->SetIndex(index);
    pane->showOnOffButton();
    pane->showVisibilityButtons();
    pane->setTitle(QString::fromLatin1(prim->idString().c_str()));
    pane->setIcon(":/icons/lightgroupsicon.png");
    pane->setWidget(currWidget);
    pane->expand();
    
    connect(pane, SIGNAL(removed(PaneWidget *)), this, SLOT(removeClipGeometry(PaneWidget *)));

    ui->clipAreaLayout->addWidget(pane);

    m_clipPanes.push_back(pane);

    // Reinsert spacer following new pane
	ui->clipAreaLayout->addItem(m_clipSpacer);
}

// ----------------------------------------------------------------------------
//  Removes a light from the active scene
// ----------------------------------------------------------------------------
void MainWindow::removeClipGeometry(PaneWidget * pane)
{
    m_clipPanes.remove(pane);
    ui->clipAreaLayout->removeWidget(pane);
    delete pane;
}

// ----------------------------------------------------------------------------
//  Flip-flops the blinking action on the log tab to indicate an error 
// ----------------------------------------------------------------------------
void MainWindow::blinkTrigger(bool active)
{
	if (active) 
    {
		m_logBlinkTimer.start(800);
		m_logBlinkState = !m_logBlinkState;

		if (m_logBlinkState) 
        {
			static const QIcon icon(":/icons/erroricon.png");
            ui->tabWidget_main->setTabIcon(filescope::logTabId, icon);
		}
		else 
        {
			static const QIcon icon(":/icons/logtabicon.png");
            ui->tabWidget_main->setTabIcon(filescope::logTabId, icon);
		}
	} 
    else 
    {
		m_logBlinkTimer.stop( );
		m_logBlinkState = false;

		static const QIcon icon(":/icons/logtabicon.png");
        ui->tabWidget_main->setTabIcon(filescope::logTabId, icon);
	}
}

// ----------------------------------------------------------------------------
//  Updates the zoom indicator icon in the main window
// ----------------------------------------------------------------------------
void MainWindow::onZoomChange(float zoomFactor)
{
    ui->label_zoom->setText(format(" %1.2f ", zoomFactor).c_str());
}

// ----------------------------------------------------------------------------
//  Copies the current frame to the clipboard
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_clipboard_clicked()
{
	m_renderView->copyToClipboard();
}

// ----------------------------------------------------------------------------
//  Opens the new light selection dialogue
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_addLight_clicked()
{ 
    static int numLights = -1; numLights++; // Light UID generator

    LightDialogue lightDialogue(numLights);
    int result = lightDialogue.exec();

    if (result)
    {
        m_activeScene.lightSet->lock();
        auto light = Light::create();
        m_activeScene.lightSet->add(light);
        addLight(light, lightDialogue.nameSelected());
        m_activeScene.lightSet->setDirty();
        m_activeScene.lightSet->unlock();
    }
}

// ----------------------------------------------------------------------------
//  Enables use of the selected device(s)
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_devicesAdd_clicked()
{
    static Qt::GlobalColor const Color_Enabled = Qt::darkGreen;

	QList<QTableWidgetItem*> items = ui->table_devices->selectedItems( );

    BOOST_FOREACH( auto & item, items )
    {
        if( item->column( ) == 0 && item->background( ) != Color_Enabled )
        {
            // Add the associated device to the renderer
            auto idItem = ui->table_devices->item( item->row( ), 0 );
            //renderer.addDevice( idItem->text( ).toInt( ) );

            // Update the items display details
            item->setBackgroundColor( Color_Enabled );
            item->setText( QString::fromUtf8("YES") );
        }
    }

	// Update device count displays
	int deviceCount = 0;
	ui->label_deviceCount_1->setText( QString("%1").arg(deviceCount) );
	ui->label_deviceCount_2->setText( QString("%1").arg(deviceCount) );
}

// ----------------------------------------------------------------------------
//  Disallows use of the selected device(s)
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_devicesRemove_clicked()
{
    static Qt::GlobalColor const Color_Disabled = Qt::darkRed;

	QList<QTableWidgetItem*> items = ui->table_devices->selectedItems( );

    BOOST_FOREACH(auto & item, items)
    {
        if( item->column( ) == 0 && item->background( ) != Color_Disabled )
        {
            // Add the associated device to the renderer
            auto idItem = ui->table_devices->item( item->row( ), 0 );
            //renderer.removeDevice( idItem->text( ).toInt( ) );

            // Update the items display details
		    item->setBackgroundColor( Color_Disabled );
		    item->setText( QString::fromUtf8("NO") );
        }
    }
		
	// Update device count displays
	int deviceCount = 0;
	ui->label_deviceCount_1->setText( QString::fromUtf8("%1").arg(deviceCount) );
	ui->label_deviceCount_2->setText( QString::fromUtf8("%1").arg(deviceCount) );
}

// ----------------------------------------------------------------------------
//  Exports the current render image to a file
// ----------------------------------------------------------------------------
void MainWindow::on_actionExport_Image_triggered()
{
    // :TODO: Detect available export types from Bitmap exporters
    String fileTypes;
    fileTypes += "PNG Image (*.png)\n";
    fileTypes += "JPEG Image (*.jpg)\n";
    fileTypes += "BMP Image (*.bmp)\n";
    fileTypes += "All Files (*)";

    QString filename = QFileDialog::getSaveFileName( 
        this, tr("Choose an image destination"), 
        m_lastOpenDir, fileTypes.c_str());

    std::string identifier(filename.toUtf8().data());
    if (identifier.empty()) return;
    if (identifier.front() != '/') identifier = '/' + identifier;
   
    m_lastOpenDir = QFileInfo(filename).absolutePath();

    m_renderView->saveImageToFile(identifier);
}

// ----------------------------------------------------------------------------
//  Opens a new volume data file for rendering
// ----------------------------------------------------------------------------
void MainWindow::on_actionOpen_triggered() 
{
    if (!canStopRendering( )) return;

    QString filename = QFileDialog::getOpenFileName( 
        this, tr("Choose a scene file to open"), 
        m_lastOpenDir, tr("Vox Scene Files (*.xml)\nPVM Volume (*.pvm)\nAll Files (*)"));

    if (!filename.isNull()) renderNewSceneFile(filename);
}

// ----------------------------------------------------------------------------
//  Opens a recent volume data file for viewing
// ----------------------------------------------------------------------------
void MainWindow::onActionOpenRecentFile() 
{
	if (!canStopRendering( )) return;

    // Get handle to sender action for file info access
	QAction *action = qobject_cast<QAction*>( sender( ) );

    // Load the file specified in the actions log
    renderNewSceneFile( action->data( ).toString( ) );
}

// ----------------------------------------------------------------------------
//  Stops the current render
// ----------------------------------------------------------------------------
void MainWindow::stopRender()
{
    m_renderController.stop();
}

// ----------------------------------------------------------------------------
//  Performs a gaussian convolution operation on the active volume data set
// ----------------------------------------------------------------------------
void MainWindow::on_actionGaussian_Filter_triggered()
{
    performFiltering([] (std::shared_ptr<Volume> volume) -> std::shared_ptr<Volume> { 
        Image3D<float> kernel(5, 5, 5);
        std::vector<float> gaussVec;
        volt::Conv::makeGaussianKernel(gaussVec, 0.75f, 5);
        for (int x = 0; x < 5; x++)
        for (int y = 0; y < 5; y++)
        for (int z = 0; z < 5; z++)
            kernel.at(x, y, z) = gaussVec[x] * gaussVec[y] * gaussVec[z];
        return volt::Conv::execute(*volume, kernel);
    });
}

// ----------------------------------------------------------------------------
//  Performs a laplacian convolution operation on the active volume data set
// ----------------------------------------------------------------------------
void MainWindow::on_actionLaplace_Filter_triggered()
{
    performFiltering([] (std::shared_ptr<Volume> volume) -> std::shared_ptr<Volume> { 
        Image3D<float> kernel;
        volt::Conv::makeLaplaceKernel(kernel);
        return volt::Conv::execute(*volume, kernel);
    });
}

// ----------------------------------------------------------------------------
//  Performs a filtering function on the active volume data set
// ----------------------------------------------------------------------------
void MainWindow::performFiltering(VolumeFilter filter)
{
    if (!m_activeScene.volume) return;

    try
    {
        m_renderController.stop();
        HistogramGenerator::instance()->stopGeneratingImages();
        m_activeScene.volume = filter(m_activeScene.volume);
    }
    catch (Error & error)
    {
        error.message = "Filtering error: " + error.message;

        VOX_LOG_EXCEPTION(Severity_Error, error);
    }
    catch (std::exception & error)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_GUI_LOG_CAT, String("Filtering error: ") + error.what());
    }

    beginRender();
}

// ----------------------------------------------------------------------------
//  Displays the about info window
// ----------------------------------------------------------------------------
void MainWindow::on_actionAbout_triggered() 
{
	AboutDialogue dialogue;
	dialogue.exec();
}

// ----------------------------------------------------------------------------
//  Exits the application without saving session data
// ----------------------------------------------------------------------------
void MainWindow::on_actionExit_triggered() 
{
	qApp->exit();
}

// ----------------------------------------------------------------------------
//  Exits the application after saving session data
// ----------------------------------------------------------------------------
void MainWindow::on_actionSave_and_Exit_triggered() 
{ 
	qApp->exit();
}

// ----------------------------------------------------------------------------
//  Returns the rendering to windowed mode
// ----------------------------------------------------------------------------
void MainWindow::on_actionNormal_Screen_triggered() 
{ 
	if (m_renderView->isFullScreen()) 
	{
		delete m_renderView;
		m_renderView = new RenderView( ui->frame_render );
		ui->frame_render_layout->addWidget( m_renderView, 0, 0, 1, 1 );
		m_renderView->show( );
	}
}

// ----------------------------------------------------------------------------
// Toggles full-screen rendering mode
// ----------------------------------------------------------------------------
void MainWindow::on_actionFull_Screen_triggered() 
{
	if (m_renderView->isFullScreen()) 
	{
		delete m_renderView;
		m_renderView = new RenderView( ui->frame_render );
		ui->frame_render_layout->addWidget( m_renderView, 0, 0, 1, 1 );
		m_renderView->show( );
	}
	else 
	{
		m_renderView->setParent(nullptr);
		m_renderView->move(pos()); // move renderview to same monitor as mainwindow
		m_renderView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		m_renderView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		m_renderView->showFullScreen();
	}
}

// ----------------------------------------------------------------------------
// Clears the log information from textEdit_log
// ----------------------------------------------------------------------------
void MainWindow::on_actionClear_Log_triggered() 
{
	ui->textEdit_log->setPlainText("");
    blinkTrigger( false );
}

// ----------------------------------------------------------------------------
// Copies the log information from textEdit_log
// ----------------------------------------------------------------------------
void MainWindow::on_actionCopy_Log_triggered() 
{
	QClipboard *clipboard = QApplication::clipboard();
	clipboard->setText(ui->textEdit_log->toPlainText());
}

// ----------------------------------------------------------------------------
// Toggles the side panel in the render tab window
// ----------------------------------------------------------------------------
void MainWindow::on_actionShow_Side_Panel_triggered(bool checked)
{
	ui->tabWidget_render->setVisible( checked );
}

// ----------------------------------------------------------------------------
// Check for user acknowledgment of new warning/error log entry
// ----------------------------------------------------------------------------
void MainWindow::on_tabWidget_main_currentChanged(int tabId)
{
    if (tabId == filescope::logTabId) 
    {
		blinkTrigger( false );
		static const QIcon icon(":/icons/logtabicon.png");
        ui->tabWidget_main->setTabIcon( filescope::logTabId, icon );
		statusMessage->setText("Checking Log Acknowledged");
	}
}

// ----------------------------------------------------------------------------
// Check for user acknowledgment of new warning/error log entry
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_stop_clicked()
{
    m_renderController.stop();

    changeRenderState(RenderState_Stopped);
}

// ----------------------------------------------------------------------------
// Check for user acknowledgment of new warning/error log entry
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_resume_clicked()
{
    if (m_renderController.isPaused())
    {
        m_renderController.unpause();

        changeRenderState(RenderState_Rendering);
    }
    else beginRender();
}

// ----------------------------------------------------------------------------
// Check for user acknowledgment of new warning/error log entry
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_pause_clicked()
{
    m_renderController.pause();

    changeRenderState(RenderState_Paused);
}

// ----------------------------------------------------------------------------
//  Opens the new light selection dialogue
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_addClip_clicked()
{ 
    static int numClips = -1; numClips++; // Clip UID generator

    ClipDialogue clipDialogue(numClips);
    int result = clipDialogue.exec();

    if (!result) { numClips--; return; } // Cancelled, decrement UID generator

    std::shared_ptr<vox::Primitive> prim;
    switch (clipDialogue.typeSelected())
    {
    case ClipType_Plane:
        prim = vox::Plane::create();
        break;

    default:
        VOX_LOG_ERROR(Error_Bug, VOX_GUI_LOG_CAT, "Invalid geometry type selection");
        return;
    }
    
    MainWindow::instance->scene().clipGeometry->add(prim);
    addClippingGeometry(prim);
}

// ----------------------------------------------------------------------------
//  Opens a file selection dialogue and attempts to load a new plugin
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_loadPlugin_clicked()
{
    QString filename = QFileDialog::getOpenFileName( 
        this, tr("Choose a scene file to open"), 
        m_lastOpenDir);
    
    if (!filename.isNull()) 
    {
        auto & pm = PluginManager::instance();

        try
        {
            auto info = pm.loadFromFile(filename.toLatin1().data());

            registerPlugin(info);
        }
        catch (Error & error)
        {
            VOX_LOG_EXCEPTION(vox::Severity_Error, error);
        }
    }
}

// ----------------------------------------------------------------------------
//  Callback from renderer with tonemapped framebuffer for interactive display
//  :TODO: implement scene locking, move this stuff outside of callback
// ----------------------------------------------------------------------------
void MainWindow::onFrameReady(std::shared_ptr<vox::FrameBuffer> frame)
{
    // Process scene interaction through the view
    m_renderView->processSceneInteractions();

    // Lock the framebuffer and issue the frameReady signal
    emit frameReady(std::make_shared<FrameBufferLock>(frame));
}

// ----------------------------------------------------------------------------
//  Exports a scene file containing scene configuration information
// ----------------------------------------------------------------------------
void MainWindow::on_actionExport_Scene_File_triggered()
{
    QString filename = QFileDialog::getSaveFileName( 
        this, tr("Choose a scene file to open"), 
        m_lastOpenDir, tr("Vox Scene File (*.xml)"));

    std::string identifier(filename.toUtf8().data());
    if (identifier.empty()) return;
    if (identifier.front() != '/') identifier = '/' + identifier;

    vox::ResourceOStream ofile(identifier);

    scene().exprt(ofile);
}