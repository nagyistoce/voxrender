/* ===========================================================================

	Project: VoxRender - Main Window Interface

	Description: Implements the main window for the VoxRender GUI

    Copyright (C) 2012-2013 Lucas Sherman

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
#include "arealightwidget.h"
#include "pointlightwidget.h"
#include "lightdialogue.h"
#include "clipdialogue.h"
#include "clipplanewidget.h"
#include "histogramgenerator.h"

// VoxRender Includes
#include "VoxLib/Core/VoxRender.h" // :TODO: Get rid of the batch incudes
#include "VoxLib/IO/ResourceHelper.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/Scene/RenderParams.h"
#include "VoxLib/Scene/PrimGroup.h"

// Qt4 Includes
#include <QtCore/QDateTime>
#include <QtCore/QTextStream>
#include <QtGui/QStandardItemModel>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <QtGui/QTextLayout>
#include <QtGui/QClipboard>
#include <QSettings>

// Singleton pointer
MainWindow* MainWindow::instance;

using namespace vox;

// Filescope namespace
namespace {
namespace filescope 
{
    const int logTabId = 4;  // Log tab index

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

    // Load the configuration file from the current directory
    try
    {
        vox::ResourceHelper::loadConfigFile("VoxRender.config");
    }
    catch (Error & error)
    {
        VOX_LOG_ERROR(error.code, "GUI", "Unable to load config file: " + error.message);
    }

    // VoxRender log configuration
    configureLoggingEnvironment();

    // Display and log the library version info and startup time
    vox::Logger::addEntry(vox::Severity_Info, vox::Error_None, "GUI", 
        vox::format("VoxRender Version: %1%", VOX_VERSION_STRING), 
        __FILE__, __LINE__);

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
            m_renderView, SLOT(setImage(std::shared_ptr<vox::FrameBufferLock>)));

    readSettings(); // Read in the application settings

    // Initialize the master renderer and register the interactive display callback
    m_renderer = vox::VolumeScatterRenderer::create();
    m_renderer->setRenderEventCallback(std::bind(&MainWindow::onFrameReady, 
        this, std::placeholders::_1));

	// Set initial render state 
    //renderNewSceneFile( "" );
}

// ----------------------------------------------------------------------------
//  Write the application settings file and terminate the core library
// ----------------------------------------------------------------------------
MainWindow::~MainWindow()
{
    writeSettings();

    HistogramGenerator::instance()->stopGeneratingImages();

    activeScene.reset();
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
        vox::Logger::addEntry(
            vox::Severity_Warning, vox::Error_System, "GUI", 
            "Unable to establish output stream to log file", 
            __FILE__, __LINE__);
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
    vox::PluginManager::instance().findAll(boost::bind(&MainWindow::registerPlugin, this, _1), false, true);
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
    // :TODO: allow custom icons in plugin specification -- pane->setIcon(":/icons/lightgroupsicon.png");
    
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
			statusMessage->setText(tr("Check Log Please")); 
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
	for( int i = 0; i < MaxRecentFiles; i++ ) 
    {
		m_recentFileActions[i] = new QAction( this );
		m_recentFileActions[i]->setVisible( false );

        // Connect action slot to open recent file to open slot
		connect(m_recentFileActions[i], SIGNAL(triggered()), 
                 this, SLOT(onActionOpenRecentFile()));
	}

    // Add the actions to the recent file submenu
	for( int i = 0; i < MaxRecentFiles; i++ ) 
    {
		ui->menuOpen_Recent->addAction( m_recentFileActions[i] );
	}
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
    // Terminate the current render
    m_renderController.stop();
    //m_renderController.reset();

    // Get the base filename for logging purposes
    std::string const file = boost::filesystem::path(
        filename.toUtf8().data()).filename().string();

    // New file loading info message
    vox::Logger::addEntry(
        vox::Severity_Info, vox::Error_None, VOX_LOG_CATEGORY, 
        vox::format("Loading Scene File: %1%", file).c_str(),
        __FILE__, __LINE__);

    // Update the status bar
	changeRenderState(RenderState_Parsing);
    
    // Compose the resource identifier for filesystem access
    std::string identifier(filename.toUtf8().data());
    if (identifier.front() != '/') identifier = '/' + identifier;

    // Load specified scene file
    try
    {
        // Attempt to parse the scene file content
        activeScene = vox::Scene::imprt(identifier);
        if (!activeScene.parameters)   activeScene.parameters   = RenderParams::create();
        if (!activeScene.clipGeometry) activeScene.clipGeometry = PrimGroup::create();
        // :TODO: Default other parameters if unspecified (immediate)
        // :TODO: Interactive option specification interface for import (long-term) ie raw volume width, height, etc

        // Synchronize the scene view
        synchronizeView();

        // Update the render state
        changeRenderState(RenderState_Rendering);

        setCurrentFile(filename); // Update window name
		
        // Initiate rendering of the new scene
        m_renderController.render(
            m_renderer,
            activeScene,
            std::numeric_limits<size_t>::max()
            );
    }
    catch (vox::Error & error)
    {
        // Append filename to error message
        error.message = vox::format(
            "Failed to load %1% [%2%]", file, 
            error.message); 

        // Log the exception
        vox::Logger::addEntry(error);
       
        // Update the render state
        changeRenderState(RenderState_Waiting);
    }
    catch (std::exception & error)
    {
        Logger::addEntry(
            Severity_Error, Error_Unknown, "GUI", 
            format("Unexpected exception loading %1%: %2%", 
                   file, error.what()).c_str(), 
            __FILE__, __LINE__);
    }

    emit sceneChanged(); // Send the scene change signal
}

// ----------------------------------------------------------------------------
// Sets the Vox GUI render state
// ----------------------------------------------------------------------------
void MainWindow::changeRenderState(RenderState state)
{
	switch (state) 
	{
		case RenderState_Waiting:
			// Waiting for input file. Most controls disabled.
			ui->pushButton_clipboard->setEnabled( true );
			ui->pushButton_resume->setEnabled( true );
			ui->pushButton_pause->setEnabled( true );
			ui->pushButton_stop->setEnabled( true );
			ui->tabWidget_render->setEnabled( true );
			ui->label_resolutionIcon->setVisible( true );
			ui->label_resolution->setVisible( true );
			ui->label_zoomIcon->setVisible( true );
			ui->label_zoom->setVisible( true );
			activityMessage->setText("Idle");
			statusProgress->setRange(0,100);
			break;
		case RenderState_Parsing:
			// Waiting for input file. Most controls disabled.
			ui->pushButton_clipboard->setEnabled( false );
			ui->pushButton_resume->setEnabled( false );
			ui->pushButton_pause->setEnabled( false );
			ui->pushButton_stop->setEnabled( false );
			ui->tabWidget_render->setEnabled( false );
			ui->label_resolutionIcon->setVisible( false );
			ui->label_resolution->setVisible( false );
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
			ui->label_resolutionIcon->setVisible( true );
			ui->label_resolution->setVisible( true );
			ui->label_zoomIcon->setVisible( true );
			ui->label_zoom->setVisible( true );
			activityMessage->setText("Rendering...");
			break;
		case RenderState_Tonemapping:
		case RenderState_Finished:
			activityMessage->setText(tr("Render is finished"));
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
	m_guiRenderState = state;
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
    camerawidget->synchronizeView();
    samplingwidget->synchronizeView();
    transferwidget->synchronizeView();

    static_cast<AmbientLightWidget*>(m_ambientPane->getWidget())->synchronizeView();

    // Remove any light panes from the previous render
    BOOST_FOREACH (auto & pane, m_lightPanes)
    {
        delete pane;
    }
    m_lightPanes.clear();

    // Create light panes for loaded lights
    BOOST_FOREACH (auto & light, activeScene.lightSet->lights())
    {
        addLight(light, "EMBEDED :TODO: NAME");
    }
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
		"Sampling", ":/icons/samplingicon.png" );
	panes[1] = new PaneWidget( ui->panesAreaContents, 
		"Camera Settings", ":/icons/cameraicon.png" );
	panes[2] = new PaneWidget( ui->panesAreaContents, 
		"Volume Histogram", ":/icons/histogramicon.png" );

	// Sampler settings widget
	samplingwidget = new SamplingWidget( panes[0] );
	panes[0]->setWidget( samplingwidget );
	ui->panesAreaLayout->addWidget( panes[0] );

	// Camera / film settings widget
	camerawidget = new CameraWidget( panes[1] );
	panes[1]->setWidget( camerawidget );
	ui->panesAreaLayout->addWidget( panes[1] );

	// Volume data histogram widget
	histogramwidget = new HistogramWidget( panes[2] );
	panes[2]->setWidget( histogramwidget );
	ui->panesAreaLayout->addWidget( panes[2] );

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
	infowidget = new InfoWidget( advpanes[0] );
	advpanes[0]->setWidget( infowidget );
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
    pane->showSoloButton();
    pane->setTitle(name);
    pane->setIcon(":/icons/lightgroupsicon.png");
    pane->setWidget(currWidget);
    pane->expand( );

    ui->lightsAreaLayout->addWidget(pane);

    m_lightPanes.push_back(pane);

    // Reinsert spacer following new pane
	ui->lightsAreaLayout->addItem( m_spacer );
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
        if (!plane) throw Error(__FILE__, __LINE__, "GUI", 
            "Error interpreting primitive :TODO:");
        currWidget = new ClipPlaneWidget(pane, plane); 
    }
    else
    {
        // :TODO: Default hideable attribute pane

        VOX_LOG_WARNING(Error_NotImplemented, "GUI", 
            format("Geometry type '%1%' unrecognized. '%2%' will not be editable.", prim->typeId(), prim->id()));

        return;
    }

    int index = m_clipPanes.size( );

    pane->SetIndex(index);
    pane->showOnOffButton();
    pane->showSoloButton();
    pane->setTitle(QString::fromLatin1(prim->id().c_str()));
    pane->setIcon(":/icons/lightgroupsicon.png");
    pane->setWidget(currWidget);
    pane->expand();

    ui->clipAreaLayout->addWidget(pane);

    m_clipPanes.push_back(pane);

    // Reinsert spacer following new pane
	ui->clipAreaLayout->addItem(m_clipSpacer);
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

    if (!result) numLights--; // Cancelled, decrement UID generator
    else addLight(activeScene.lightSet->addLight(), lightDialogue.nameSelected( ));
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
//  Opens a new volume data file for rendering
// ----------------------------------------------------------------------------
void MainWindow::on_actionOpen_triggered() 
{
    if (!canStopRendering( )) return;

    QString filename = QFileDialog::getOpenFileName( 
        this, tr("Choose a scene file to open"), 
        m_lastOpenDir, tr("Vox Scene Files (*.xml)"));

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
	qApp->exit( );
}

// ----------------------------------------------------------------------------
//  Exits the application after saving session data
// ----------------------------------------------------------------------------
void MainWindow::on_actionSave_and_Exit_triggered() 
{ 
	qApp->exit( );
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
		//connect(renderView, SIGNAL(viewChanged()), this, SLOT(viewportChanged())); // reconnect
		//renderView->reload( );
		m_renderView->show( );
		//ui->action_normalScreen->setEnabled( false );
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
		//connect(renderView, SIGNAL(viewChanged()), this, SLOT(viewportChanged())); // reconnect
		//renderView->reload( );
		m_renderView->show( );
		//ui->action_normalScreen->setEnabled( false );
	}
	else 
	{
		m_renderView->setParent( nullptr );
		m_renderView->move( pos() ); // move renderview to same monitor as mainwindow
		m_renderView->setHorizontalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
		m_renderView->setVerticalScrollBarPolicy( Qt::ScrollBarAlwaysOff );
		m_renderView->showFullScreen( );
		//ui->action_normalScreen->setEnabled( true );
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
    if( tabId == filescope::logTabId ) 
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
    m_renderController.unpause();

    changeRenderState(RenderState_Rendering);
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
        VOX_LOG_ERROR(Error_Bug, "GUI", "Invalid geometry type selection");
        return;
    }

    addClippingGeometry(prim);
}

// ----------------------------------------------------------------------------
// Flag to perform an image update for the next render callback
// ----------------------------------------------------------------------------
void MainWindow::on_pushButton_imagingApply_clicked()
{
    m_imagingUpdate = true;
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
//  :TODO: Need to seriously add some thread locking around this stuff
//         and the transfer function stuff too
// ----------------------------------------------------------------------------
void MainWindow::onFrameReady(std::shared_ptr<vox::FrameBuffer> frame)
{
    // Process scene interaction through the view
    m_renderView->processSceneInteractions();

    // Process scene interactions through the interface
    if (ui->checkBox_imagingAuto->isChecked() || m_imagingUpdate)
    {
        m_imagingUpdate = false; // Reset the image update flag

        // Process any changes to the lighting control widgets
        BOOST_FOREACH (auto & pane, m_lightPanes)
        {
            auto widget = static_cast<PointLightWidget*>(pane->getWidget());
            widget->processInteractions();
        }
               
        // Process any changes to the clip geometry
        BOOST_FOREACH (auto & pane, m_clipPanes)
        {
            auto widget = static_cast<ClipPlaneWidget*>(pane->getWidget());
            widget->processInteractions();
        }

        static_cast<AmbientLightWidget*>(m_ambientPane->getWidget())->processInteractions();
        camerawidget->processInteractions();
        samplingwidget->processInteractions();
    }

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