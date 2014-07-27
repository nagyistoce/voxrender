/* ===========================================================================

	Project: VoxRender

	Description: Implements an interface for transfer modification.

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
#include "transferwidget.h"
#include "ui_transferwidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"
#include "Actions/MaterialEditAct.h"

// VoxLib Dependencies
#include "VoxLib/Action/ActionManager.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Logging.h"

// QT Includes
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>

// Transfer function modification wrapper for auto-update
#define DO_LOCK(X)                              \
    auto scene = MainWindow::instance->scene(); \
    auto lock = scene->lock(this);              \
    X;                                          \
    m_transfer->setDirty();                     \
    lock.reset(); 

using namespace vox;

// File scope namespace
namespace {
namespace filescope {
    
    // ----------------------------------------------------------------------------
    //  Converts a Vec3 Color RGB normalized to a QColor
    // ----------------------------------------------------------------------------
    QColor toQColor(Vector<UInt8,3> rgbColor)
    {
        return QColor( rgbColor[0], rgbColor[1], rgbColor[2]);
    }

} // namespace filescope
} // namespace anonymous

// ----------------------------------------------------------------------------
// Constructor - Initialize widget slots and signals
// ----------------------------------------------------------------------------
TransferWidget::TransferWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TransferWidget),
    m_colorDiffuse(new QColorPushButton()),
    m_colorEmissive(new QColorPushButton()),
    m_colorSpecular(new QColorPushButton()),
    m_ignore(false)
{
    ui->setupUi(this);

	// Transfer function view elements (with histogram underlay)
	m_primaryView   = new HistogramView(ui->transferPrimary,   true);
	m_secondaryView = new HistogramView(ui->transferSecondary, true);
	ui->gridLayout_transferPrimary->addWidget(m_primaryView, 0, 0, 1, 1);
	ui->gridLayout_transferSecondary->addWidget(m_secondaryView, 0, 0, 1, 1);

    // Add the color selection widgets to the layout
    ui->layout_diffuseColor->addWidget(m_colorDiffuse);
    ui->layout_emissiveColor->addWidget(m_colorEmissive);
    ui->layout_specularColor->addWidget(m_colorSpecular);
    connect(m_colorDiffuse,  SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorDiffuseChanged(const QColor&)));
    connect(m_colorEmissive, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorEmissiveChanged(const QColor&)));
    connect(m_colorSpecular, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorSpecularChanged(const QColor&)));
    connect(m_colorDiffuse,  SIGNAL(beginColorSelection()), this, SLOT(beginMaterialChange()));
    connect(m_colorEmissive, SIGNAL(beginColorSelection()), this, SLOT(beginMaterialChange()));
    connect(m_colorSpecular, SIGNAL(beginColorSelection()), this, SLOT(beginMaterialChange()));
    connect(m_colorDiffuse,  SIGNAL(endColorSelection()), this, SLOT(endMaterialChange()));
    connect(m_colorEmissive, SIGNAL(endColorSelection()), this, SLOT(endMaterialChange()));
    connect(m_colorSpecular, SIGNAL(endColorSelection()), this, SLOT(endMaterialChange()));

    connect(MainWindow::instance, SIGNAL(sceneChanged(vox::Scene &,void *)), 
            this, SLOT(sceneChanged(vox::Scene &,void *)), Qt::DirectConnection);

    setSelectedNode(nullptr); // Initialize the widget to no curr node settings
}
    
// ----------------------------------------------------------------------------
//  Destructor - Delete transfer views
// ----------------------------------------------------------------------------
TransferWidget::~TransferWidget()
{
    delete ui;
}

// ----------------------------------------------------------------------------
//  Returns the currently selected transfer function node
// ----------------------------------------------------------------------------
std::shared_ptr<Node> TransferWidget::selectedNode()
{
    return m_currentNode;
}

// ----------------------------------------------------------------------------
//  Returns the currently selected transfer function node
// ----------------------------------------------------------------------------
std::shared_ptr<Quad> TransferWidget::selectedQuad(Quad::Node * node)
{
    if (node) *node = m_currentIndex;

    return m_currentQuad;
}

// ----------------------------------------------------------------------------
//  Sets the selected node for 1D transfer function editing
// ----------------------------------------------------------------------------
void TransferWidget::setSelectedNode(std::shared_ptr<vox::Node> node, bool signal)
{
    m_currentNode = node;
    
    m_ignore = true;

    if (node)
    {
        setSelectedMaterial(m_currentNode->material);
        
        // Enable al off the active node controls
        ui->groupBox_currNode->setDisabled(false);
        ui->groupBox_nodePos->setDisabled(false);
        
        // Update all of the active node controls
        ui->doubleSpinBox_density->setValue(m_currentNode->density*100.0f);
        ui->checkBox_visible->setChecked(true);
    }
    else
    {
        setSelectedMaterial(nullptr);

        // Disable all of the active node controls
        ui->groupBox_currNode->setDisabled(true);
        ui->groupBox_nodePos->setDisabled(true);
    }
    
    if (signal) emit nodeChanged(EditType_Reference);

    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Sets the selected quad for 2D transfer function editing
// ----------------------------------------------------------------------------
void TransferWidget::setSelectedQuad(std::shared_ptr<Quad> quad, Quad::Node node, bool signal)
{
    m_ignore = true;

    // Set the selected material within the quad
    if (!quad || node == Quad::Node::Node_End) 
        setSelectedMaterial(nullptr);
    else setSelectedMaterial(quad->materials[node]);

    // Set the node controls for the selected component of the quad
    m_currentQuad = quad;
    m_currentIndex = node;

    if (signal) emit nodeChanged(EditType_Reference);
    
    m_ignore = false;
}

// ----------------------------------------------------------------------------
//  Sets the selected material for transfer function editing
// ----------------------------------------------------------------------------
void TransferWidget::setSelectedMaterial(std::shared_ptr<Material> material)
{
    m_currentMaterial = material;

    if (material)
    {
        ui->groupBox_material->setDisabled(false);
        ui->doubleSpinBox_gloss->setValue(material->glossiness*100.0f);
        ui->doubleSpinBox_opacity->setValue(material->opticalThickness*100.0f);

        m_colorDiffuse->setColor(filescope::toQColor(material->diffuse), true);
        m_colorEmissive->setColor(filescope::toQColor(material->emissive), true);
        m_colorSpecular->setColor(filescope::toQColor(material->specular), true);
    }
    else ui->groupBox_material->setDisabled(true); 
}

// ----------------------------------------------------------------------------
//  Processes and re-emits a node change event
// ----------------------------------------------------------------------------
void TransferWidget::onNodeChanged()
{
    // Reload the node into the widgets
    if      (m_currentNode) setSelectedNode(m_currentNode, false);
    else if (m_currentQuad) setSelectedQuad(m_currentQuad, Quad::Node_End, false);

    emit nodeChanged();
}

// ----------------------------------------------------------------------------
//  Synchronizes the transfer function widget with the active
// ----------------------------------------------------------------------------
void TransferWidget::sceneChanged(Scene & scene, void * userInfo)
{
    if (userInfo == this || !scene.transfer) return;
    m_transfer = scene.transfer;

    m_ignore = true;

    auto res = m_transfer->resolution();
    ui->spinBox_resX->setValue(res[0]);
    ui->spinBox_resY->setValue(res[1]);
    ui->spinBox_resZ->setValue(res[2]);

    m_ignore = false;

    if (auto transfer1D = dynamic_cast<Transfer1D*>(m_transfer.get()))
    {
        switchDimensions(1);
    
        auto nodes = transfer1D->nodes();
        if (std::find(nodes.begin(), nodes.end(), m_currentNode) == nodes.end()) m_currentNode = nullptr;
        m_currentQuad = nullptr;
        setSelectedNode(m_currentNode);
    }
    else if (auto transfer2D = dynamic_cast<Transfer2D*>(m_transfer.get()))
    {
        switchDimensions(2);

        auto nodes = transfer2D->quads();
        m_currentNode = nullptr;
        if (std::find(nodes.begin(), nodes.end(), m_currentQuad) == nodes.end()) m_currentQuad = nullptr;
        setSelectedQuad(m_currentQuad);
    }
}

// ----------------------------------------------------------------------------
//  Tracks scene interaction through key event input
// ----------------------------------------------------------------------------
void TransferWidget::keyPressEvent(QKeyEvent * event)
{
    switch (event->key())
    {
    case Qt::Key_Delete: on_pushButton_delete_clicked(); break;
    }
}

// ----------------------------------------------------------------------------
// Returns true if the user allows dimension change
// ----------------------------------------------------------------------------
bool TransferWidget::canSwitchDimensions()
{
	QMessageBox msgBox(this);
	msgBox.setIcon(QMessageBox::Question);
	msgBox.setText(tr("Are you sure you want to switch dimension?\
                   Any unsaved changes to the current function will be lost."));
	msgBox.setWindowTitle("Switch Dimensions");

	QPushButton* accept = msgBox.addButton(tr("Yes"), QMessageBox::AcceptRole);
	msgBox.addButton(tr("No"), QMessageBox::RejectRole);
	QPushButton* cancel = msgBox.addButton(tr("Cancel"), QMessageBox::RejectRole);
	msgBox.setDefaultButton(cancel);

	msgBox.exec();

	if (msgBox.clickedButton() != accept) return false;

	return true;
}

// ----------------------------------------------------------------------------
// Blocks radio button signals 
// ----------------------------------------------------------------------------
void TransferWidget::switchDimensions(int nDims)
{
    if (nDims == m_dimensions) return;

    setSelectedNode(nullptr, false);
    setSelectedQuad(nullptr, Quad::Node_End, false);

	// Block radio button signals
	ui->radioButton_1->blockSignals(true);
	ui->radioButton_2->blockSignals(true);
	ui->radioButton_3->blockSignals(true);

	// Switch transfer control context
	switch (nDims) 
	{
		case 1:

			// Configure widget for 1-Dimensional work 
            if (!dynamic_cast<Transfer1D*>(m_transfer.get())) m_transfer = Transfer1D::create();
            m_primaryView->setType(HistogramView::DataType_Density);
			ui->radioButton_1->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->hide( );
			ui->doubleSpinBox_gradient->hide( );
			ui->label_gradient->hide( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
            ui->spinBox_resX->setEnabled(true);
            ui->spinBox_resY->setEnabled(false);
            ui->spinBox_resZ->setEnabled(false);
			break;

		case 2:

			// Configure widget for 2-Dimensional work
            if (!dynamic_cast<Transfer2D*>(m_transfer.get())) m_transfer = Transfer2D::create();
            m_primaryView->setType(HistogramView::DataType_DensityGrad);
			ui->radioButton_2->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
            ui->spinBox_resX->setEnabled(true);
            ui->spinBox_resY->setEnabled(true);
            ui->spinBox_resZ->setEnabled(false);
			break;

		case 3:

			// Configure widget for 3-Dimensional work
            m_primaryView->setType(HistogramView::DataType_DensityGrad);
            m_secondaryView->setType(HistogramView::DataType_DensityLap);
			ui->radioButton_3->setChecked(true); 
			ui->groupBox_transferSecondary->show();
			ui->horizontalSlider_gradient->show();
			ui->doubleSpinBox_gradient->show();
			ui->label_gradient->show();
			ui->horizontalSlider_gradient2->show();
			ui->doubleSpinBox_gradient2->show();
			ui->label_gradient2->show();
            ui->spinBox_resX->setEnabled(true);
            ui->spinBox_resY->setEnabled(true);
            ui->spinBox_resZ->setEnabled(true);
			break; 

        default:
            break;
	}

    m_dimensions = nDims;

    // Update the scene
    if (m_transfer != MainWindow::instance->scene()->transfer)
    {
        auto main = MainWindow::instance;
        auto lock = main->scene()->lock();
        main->scene()->transfer = m_transfer;
        main->m_renderController.setTransferFunction(m_transfer); 
        // :TODO: Some sort of scene transfer swapping functionality, this should NOT be through render controller
        //        this sort of effect may be useful for clipping configurations at a later point as well.
    }

	// Unblock radio button signals
	ui->radioButton_1->blockSignals(false);
	ui->radioButton_2->blockSignals(false);
	ui->radioButton_3->blockSignals(false);
}

// ----------------------------------------------------------------------------
// Single dimension transfer selection
// ----------------------------------------------------------------------------
void TransferWidget::on_radioButton_1_toggled(bool checked)
{
	if (checked && canSwitchDimensions()) switchDimensions(1);
}

// ----------------------------------------------------------------------------
// Single dimension transfer selection
// ----------------------------------------------------------------------------
void TransferWidget::on_radioButton_2_toggled(bool checked)
{
	if (checked && canSwitchDimensions()) switchDimensions(2);
}

// ----------------------------------------------------------------------------
// Single dimension transfer selection
// ----------------------------------------------------------------------------
void TransferWidget::on_radioButton_3_toggled(bool checked)
{	
	if (checked && canSwitchDimensions()) switchDimensions(3);
}

// ----------------------------------------------------------------------------
//  Opens a dialogue to import a transfer function from a user defined file
// ----------------------------------------------------------------------------
void TransferWidget::load()
{
    QString filename = QFileDialog::getOpenFileName( 
        this, tr("Choose a scene file to open"), 
        MainWindow::instance->lastOpenDir(), 
        tr("Vox Scene Files (*.xml)"));
    
    if (filename.isNull()) return;

    QFileInfo info(filename);
    MainWindow::instance->setLastOpenDir(info.absolutePath());

    // Compose the resource identifier for filesystem access
    std::string identifier(filename.toUtf8().data());
    if (identifier.front() != '/') identifier = '/' + identifier;

    // Attempt to parse the scene file 
    vox::OptionSet options;
    options.addOption("ImportVolume",  false);
    options.addOption("ImportCamera",  false);
    options.addOption("ImportLights",  false);
    options.addOption("ImportClipGeo", false);
    options.addOption("ImportParams",  false);
    auto scene = vox::Scene::imprt(identifier, options);

    // Update the scene and restart the render
    auto activeScene = MainWindow::instance->scene();
    auto lock = activeScene->lock();
    m_transfer = scene->transfer;
    activeScene->transfer = scene->transfer;
    MainWindow::instance->m_renderController.setTransferFunction(m_transfer); 
    lock.reset();
}

// ----------------------------------------------------------------------------
//  Opens a dialogue to export a transfer function to a user defined file
// ----------------------------------------------------------------------------
void TransferWidget::save()
{
    try
    {
        // Get a filename from the user
        QString filename = QFileDialog::getSaveFileName( 
            this, tr("Choose a file location"), 
            MainWindow::instance->lastOpenDir(), 
            "Vox Scene Files (*.xml)\n"
            "Raw Image Files (*.bmp with xml)");

        if (filename.isNull()) return;
    
        QFileInfo info(filename);
        MainWindow::instance->setLastOpenDir(info.absolutePath());

        // Compose the resource identifier for filesystem access
        std::string idStr(filename.toUtf8().data());
        if (idStr.front() != '/') idStr = '/' + idStr;
        ResourceId identifier(idStr);

        // Generate the export options for a transfer function
        vox::OptionSet options;
        options.addOption("ExportVolume",    false);
        options.addOption("ExportCamera",    false);
        options.addOption("ExportLights",    false);
        options.addOption("ExportClipGeo",   false);
        options.addOption("ExportParams",    false);
        options.addOption("ExportAnimation", false);
        auto extension = identifier.extractFileExtension();
        if (extension != ".xml")
        {
            options.addOption("ForceTransferMap", extension);
            identifier.setFileExtension(".xml");
        }

        // Export the transfer function as specified
        MainWindow::instance->scene()->exprt(identifier, options);
    }
    catch (Error & error)
    {
        VOX_LOG_EXCEPTION(Severity_Error, error);
    }
    catch (...)
    {
        VOX_LOG_ERROR(Error_Unknown, VOX_LOG_CATEGORY, "Error exporting scene file");
    }
}

// ----------------------------------------------------------------------------
// Select the first node / previous region
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_first_clicked()
{
    if (!m_transfer) return;

    if (auto transfer1D = dynamic_cast<Transfer1D*>(m_transfer.get()))
    {
        setSelectedNode(transfer1D->nodes().front());
    }
}

// ----------------------------------------------------------------------------
//  Select the next node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_next_clicked()
{
    if (m_currentNode)
    {
        if (auto transfer1D = dynamic_cast<Transfer1D*>(m_transfer.get()))
        {
            // Check if this is the last node
            auto nodes = transfer1D->nodes();
            if (m_currentNode != nodes.back())
            {
                // Acquire a handle to the subsequent node in the linked list
                auto iter = std::find(nodes.begin(), nodes.end(), m_currentNode);
                setSelectedNode(*(++iter));
            }
        }
    }
}

// ----------------------------------------------------------------------------
//  Select the previous node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_prev_clicked()
{
    if (!m_currentNode) return;

    if (auto transfer1D = dynamic_cast<Transfer1D*>(m_transfer.get()))
    {
        // Check if this is the first node
        auto nodes = transfer1D->nodes();
        auto iter  = std::find(nodes.begin(), nodes.end(), m_currentNode);
        if (iter != nodes.begin())
        {
            // Acquire a handle to the previous node in the linked list
            setSelectedNode(*(--iter));
        }
    }
}

// ----------------------------------------------------------------------------
//  Select the last node / next region
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_last_clicked()
{
    if (!m_transfer) return;

    if (auto transfer1D = dynamic_cast<Transfer1D*>(m_transfer.get()))
    {
        setSelectedNode(transfer1D->nodes().back());
    }
}

// ----------------------------------------------------------------------------
//  Delete selected node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_delete_clicked()
{
    if (!m_currentNode && !m_currentQuad) return;

    auto scene = MainWindow::instance->scene();
    auto lock = scene->lock();

    //emit nodeChanged(EditType_Remove);

    if (auto transfer1D = dynamic_cast<Transfer1D*>(m_transfer.get()))
    {
        transfer1D->remove(m_currentNode);
    }
    else if (auto transfer2D = dynamic_cast<Transfer2D*>(m_transfer.get()))
    {
        transfer2D->remove(m_currentQuad);
    }

    m_transfer->setDirty();
}

// ----------------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_density_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_density,
        ui->horizontalSlider_density,
        value);
    
    if (m_ignore) return;

    auto val = ui->doubleSpinBox_density->value() / 100.0f;
    DO_LOCK(m_currentNode->density = val; emit nodeChanged();)
}

// ----------------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_density_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_density,
        ui->doubleSpinBox_density,
        value);

    if (m_ignore) return;

    auto val = ui->doubleSpinBox_density->value() / 100.0f;
    DO_LOCK(m_currentNode->density = val; emit nodeChanged();)
}

// ----------------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_gloss_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_gloss,
        ui->horizontalSlider_gloss,
        value);
    
    if (m_ignore) return;

    auto val = ui->doubleSpinBox_gloss->value() / 100.0f;
    DO_LOCK(m_currentMaterial->glossiness = val;)
}

// ----------------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_gloss_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_gloss,
        ui->doubleSpinBox_gloss,
        value);

    if (m_ignore) return;

    auto val = ui->doubleSpinBox_gloss->value() / 100.0f;
    DO_LOCK(m_currentMaterial->glossiness = val;)
}

// ----------------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_opacity_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_opacity,
        ui->horizontalSlider_opacity,
        value);

    if (m_ignore) return;

    auto val = ui->doubleSpinBox_opacity->value() / 100.0f;
    DO_LOCK(m_currentMaterial->opticalThickness = val; emit nodeChanged();)
}

// ----------------------------------------------------------------------------
//  Modifies the opacity of the transfer node
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_opacity_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_opacity,
        ui->doubleSpinBox_opacity,
        value);

    if (m_ignore) return;

    auto val = ui->doubleSpinBox_opacity->value() / 100.0f;
    DO_LOCK(m_currentMaterial->opticalThickness = val; emit nodeChanged();)
}

// ----------------------------------------------------------------------------
//  Modifies the emissive strength of the transfer node
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_emissiveStr_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_emissiveStr,
        ui->horizontalSlider_emissiveStr,
        value);

    if (m_ignore) return;

    DO_LOCK(m_currentMaterial->emissiveStrength = ui->doubleSpinBox_emissiveStr->value();)
}

// ----------------------------------------------------------------------------
//  Modifies the emissive strength of the transfer node
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_emissiveStr_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_emissiveStr,
        ui->doubleSpinBox_emissiveStr,
        value);

    if (m_ignore) return;

    DO_LOCK(m_currentMaterial->emissiveStrength = ui->doubleSpinBox_emissiveStr->value();)
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void TransferWidget::colorDiffuseChanged(QColor const& color)
{
    if (m_ignore) return;
    auto cast = Vector<UInt8,3>(color.red(), color.green(), color.blue());
    DO_LOCK(m_currentMaterial->diffuse = cast;)
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void TransferWidget::colorEmissiveChanged(QColor const& color)
{
    if (m_ignore) return;
    auto cast = Vector<UInt8,3>(color.red(), color.green(), color.blue());
    DO_LOCK(m_currentMaterial->emissive = cast;)
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void TransferWidget::colorSpecularChanged(QColor const& color)
{
    if (m_ignore) return;
    auto cast = Vector<UInt8,3>(color.red(), color.green(), color.blue());
    DO_LOCK(m_currentMaterial->specular = cast;)
}

// --------------------------------------------------------------------
//  Registers an actionable event when a material change is cemented
// --------------------------------------------------------------------
void TransferWidget::beginMaterialChange()
{
    m_editMaterial = Material::create();
    *m_editMaterial = *m_currentMaterial;
}

// --------------------------------------------------------------------
//  Registers an actionable event when a material change is cemented
// --------------------------------------------------------------------
void TransferWidget::endMaterialChange()
{
    if (!m_currentMaterial || !m_editMaterial) return;
    if (*m_currentMaterial != *m_editMaterial)
        ActionManager::instance().push(std::make_shared<MaterialEditAct>(m_currentMaterial, m_editMaterial));
    m_editMaterial.reset();
}

// --------------------------------------------------------------------
//  Transfer function resolution
// --------------------------------------------------------------------
void TransferWidget::on_spinBox_resX_valueChanged(int value) { updateResolution(); }
void TransferWidget::on_spinBox_resY_valueChanged(int value) { updateResolution(); }
void TransferWidget::on_spinBox_resZ_valueChanged(int value) { updateResolution(); }
void TransferWidget::updateResolution()
{
    if (m_ignore) return;

    DO_LOCK(m_transfer->setResolution(Vector3u(ui->spinBox_resX->value(), ui->spinBox_resY->value(), ui->spinBox_resZ->value())))
}