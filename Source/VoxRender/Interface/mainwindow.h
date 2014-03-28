/* ===========================================================================

	Project: VoxRender

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

// Begin definition
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// Include Dependencies
#include "VoxScene/RenderController.h"
#include "VoxScene/Scene.h"
#include "VoxScene/Light.h"
#include "VoxScene/Transfer.h"
#include "VoxScene/FrameBuffer.h"
#include "VoxScene/Primitive.h"
#include "VoxLib/Plugin/PluginManager.h"
#include "VoxLib/Core/Logging.h"
#include "VoxVolt/Filter.h"

// Standard Renderers for the Application
#include "VolumeScatterRenderer/Core/VolumeScatterRenderer.h"

// Include Dependencies
#include "histogramwidget.h"
#include "panewidget.h"
#include "renderview.h"
#include "samplingwidget.h"
#include "transferwidget.h"

// QT Includes
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMainWindow.h>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QProgressDialog>
#include <QtWidgets/QSpacerItem>
#include <QtCore/QTimer.h>

/** GUI Render States */
enum RenderState
{
	RenderState_Waiting,		///< Idling
	RenderState_Loading,		///< Loading Scene File
	RenderState_Rendering,		///< Rendering Scene
    RenderState_Animating,      ///< Rendering Animation
	RenderState_Stopping,		///< Stopping Rendering
	RenderState_Stopped,		///< Stopped Rendering
	RenderState_Paused,			///< Paused Rendering
};

namespace vox { namespace volt { class Filter; } }

// Generated class
namespace Ui { class MainWindow; }

// Forward declarations
class AmbientLightWidget;
class PluginWidget;
class CameraWidget;
class PointLightWidget;
class InfoWidget;
class TimingWidget;
class AnimateWidget;

// Convenience typedef for a volume filtering function
typedef std::function<std::shared_ptr<vox::Volume>(std::shared_ptr<vox::Volume>)> VolumeFilter;

// Shared pointer MetaType declaration for QT signals/slots
Q_DECLARE_METATYPE(std::shared_ptr<vox::FrameBufferLock>);
Q_DECLARE_METATYPE(std::string);
Q_DECLARE_METATYPE(std::shared_ptr<vox::Node>);
Q_DECLARE_METATYPE(std::shared_ptr<vox::volt::Filter>);

// Main Application Window
class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);

    ~MainWindow();
    
    static MainWindow* instance;
    
    /** Adds a control for managing a clipping object in the render scene */
    void addClippingGeometry(std::shared_ptr<vox::Primitive> prim);

    /** Returns the current render state of the application */
    RenderState renderState() { return m_guiRenderState; }

    /** Current scene accessor */
    vox::Scene & scene() { return m_activeScene; }

    /** Returns the currently active renderer */ 
    vox::VolumeScatterRenderer & renderer() { return *m_renderer; }

    /** Returns the transfer widget object handle */
    TransferWidget * transferWidget() { return transferwidget; }

    /** Sets the working transfer node for global editing */
    // :TODO: Move to TransferWidget class
    void setTransferNode(std::shared_ptr<vox::Node> node)
    {
        emit transferNodeSelected(node);
    }
    void setTransferQuad(std::shared_ptr<vox::Quad> quad, vox::Quad::Node node)
    {
        emit transferQuadSelected(quad, node);
    }

    /** Returns the last directory accessed by the user */
    QString const& lastOpenDir() { return m_lastOpenDir; }

    /** Updates the last directory accessed by the user */
    void setLastOpenDir(QString const& lastOpenDir) { m_lastOpenDir = lastOpenDir; } 

	vox::RenderController m_renderController; ///< Application render controller

    std::shared_ptr<vox::VolumeScatterRenderer> m_renderer; ///< CUDA device renderer

	InfoWidget* m_infowidget; ///< Advanced info widget
    
    /** Initiates rendering operations for the scene (or the animator, if specified) */
    void beginRender(size_t samples = std::numeric_limits<size_t>::max(), bool animation = false);

    /** Stops the current rendering operation */
    void stopRender();

signals:
    /** Signal sent when the active scene is reloaded */
    void sceneChanged();

    /** Progress change callback */
    void progressChanged(int progress);

    /** Signal sent when the working transfer node is changed */
    void transferNodeSelected(std::shared_ptr<vox::Node> node); 
    void transferQuadSelected(std::shared_ptr<vox::Quad> quad, vox::Quad::Node node); 

    /** Signal sent when the RenderController feeds back a frame */
    void frameReady(std::shared_ptr<vox::FrameBufferLock> frame);

    /** Signals log entry recieval for pickup by the GUI thread */
    void logEntrySignal(char const* file, int line, int severity, 
        int code, char const* category, std::string message);

    /** Change event callback for available filters */
    void filtersChanged(std::shared_ptr<vox::volt::Filter> filter, bool available);

private:
    Ui::MainWindow *ui;
    
    /** Vox GUI Error Handler - Logs to std::clog and log tab pane */
    void voxGuiErrorHandler(char const* file, int line, int severity, 
        int code, char const* category, char const* message);

    /** Configures the VoxRender logging system */
    void configureLoggingEnvironment();

    /** Configures the available plugins */
    void configurePlugins();

    /** Adds a newly detected plugin to the available plugins */
    void registerPlugin(std::shared_ptr<vox::PluginInfo> plugin);

    /** Sets the current scene file display name */
    void setCurrentFile(const QString& path);
    
    // Frame Ready Callback
    void onFrameReady(std::shared_ptr<vox::FrameBuffer> frame);

	// Render state control
	RenderState m_guiRenderState;
    void renderNewSceneFile( const QString& filename );
	void changeRenderState( RenderState state );
	bool canStopRendering();
    void synchronizeView();

    /** Helper method for executing a volume filtering operation */
    void performFiltering(std::shared_ptr<vox::volt::Filter> filter, vox::OptionSet const& options);

	// Render status bar
	QLabel       * activityLabel;   ///< "activity" label
	QLabel       * statusLabel;     ///< "status" label
	QLabel       * statsLabel;      ///< "statistics" label
	QLabel       * activityMessage; ///< Name of activity state
	QLabel       * statusMessage;   ///< Description of render status
	QProgressBar * statusProgress;  ///< Progress of current activity
	QLabel       * statsMessage;    ///< Statistics label
	QSpacerItem  * spacer;          ///< Spacer for status bar

	RenderView * m_renderView; ///< View panel for current render

	// File / Directory Info
	enum { MaxRecentFiles = 5 };
	QAction* m_recentFileActions[MaxRecentFiles];
    void createRecentFileActions();
    void updateRecentFileActions();
	QList<QFileInfo> m_recentFiles;
	QString m_lastOpenDir;

	// Window initialization 
    void createDeviceTable();
    void createRenderTabPanes();
    void createStatusBar();
    void writeSettings();
    void readSettings();

	// Render tab panes
	enum { NumPanes = 5 };
	PaneWidget      * panes[NumPanes]; ///< Render tab widget panes
	HistogramWidget * histogramwidget; ///< Histogram view widget
	SamplingWidget  * samplingwidget;  ///< Sampling parameters widget
	CameraWidget    * camerawidget;    ///< Camera settings widget
    TimingWidget    * timingwidget;    ///< Volume time index widget
    AnimateWidget   * animationwidget; ///< Animation control widget

	enum { NumAdvPanes = 1 };
	PaneWidget* advpanes[NumAdvPanes];  ///< Advanced tab widget panes

	enum { NumTransferPanes = 0 };
	TransferWidget* transferwidget;
    
    // -------------------------------------------
    // :TODO: CHUNK OF STUFF TO BE MOVED TO SEPERATE WIDGET
    // Light panel panes
    void addLight(std::shared_ptr<vox::Light> light);
    void removeLight(std::shared_ptr<vox::Light> light);
    PaneWidget *           m_ambientPane;
	std::list<PaneWidget*> m_lightPanes;
    QSpacerItem *          m_spacer;
    // -------------------------------------------
    
    // -------------------------------------------
    // :TODO: CHUNK OF STUFF TO BE MOVED TO SEPERATE WIDGET
    // Clipping Geometry panel panes
    void removeClipGeometry(std::shared_ptr<vox::Primitive> prim);
	std::list<PaneWidget*> m_clipPanes;
    QSpacerItem *          m_clipSpacer;
    // -------------------------------------------

    // Plugin panes
    QVector<PaneWidget*> m_pluginPanes;
    QSpacerItem *        m_pluginSpacer;
    
	vox::Scene m_activeScene; ///< Current scene

    // --------------------------------------------------------------------
    //  Do not place anything below here, the log stream must be closed 
    //  after all other object have been shutdown 
    // --------------------------------------------------------------------

    std::ofstream m_logFileStream;  ///< Output stream for logging to disk
    QTimer        m_logBlinkTimer;  ///< Timer for blinking log tab icon
    bool          m_logBlinkState;  ///< Log blink flip-flop state

private slots:
	// Render pushbuttons
	void on_pushButton_clipboard_clicked();
    void on_pushButton_addLight_clicked();
    void on_pushButton_addClip_clicked();

	// Toolbar action slots
    void on_actionOverlay_Statistics_triggered(bool checked);
    void on_actionFull_Screen_triggered();
    void on_actionOpen_triggered();
    void on_actionAbout_triggered();
    void on_actionExit_triggered();
    void on_actionUndo_triggered();
    void on_actionRedo_triggered();
    void on_actionSave_and_Exit_triggered();
    void on_actionNormal_Screen_triggered();
    void on_actionClear_Log_triggered();
    void on_actionCopy_Log_triggered();
	void on_actionShow_Side_Panel_triggered(bool checked);
    void on_actionExport_Image_triggered();
    void onActionOpenRecentFile();
    void on_actionExport_Scene_File_triggered();

    void on_pushButton_stop_clicked();
    void on_pushButton_resume_clicked();
    void on_pushButton_pause_clicked();

    void on_pushButton_refreshPlugins_clicked() { }
    void on_pushButton_loadPlugin_clicked();

    void onZoomChange(float zoom);

    void onProgressChanged(int progress);

    // Filter action slot
    void onFilterExecuted();

    // Scene element removal
    void removeClipGeometry(PaneWidget * pane);
    void removeLight(PaneWidget * pane);

	// Device selection action slots
	void on_pushButton_devicesAdd_clicked();
	void on_pushButton_devicesRemove_clicked();

    /** VoxLib Error Handler which prints log entries to the main window log tab */
    void printLogEntry(char const* file, int line, int severity, int code, 
                       char const* category, std::string message);
    
    /** Change event handler for available scene filters */
    void onFiltersChanged(std::shared_ptr<vox::volt::Filter> filter, bool available);

    // Log tab widget catch for detecting log checks
    void on_tabWidget_main_currentChanged(int tabId);
    void blinkTrigger(bool active = true);
};

#endif // MAINWINDOW_H
