/* ===========================================================================

	Project: TransferWidget - Transfer Function Widget

	Description:
	 Implements an interface for interactive transfer function modification.

    Copyright (C) 2012 Lucas Sherman

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

// VoxLib Dependencies
#include "VoxLib/Core/Geometry/Vector.h"

// QT Includes
#include <QtWidgets/QMessageBox>

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
    m_colorSpecular(new QColorPushButton())
{
    ui->setupUi(this);

	// Transfer function view elements (with histogram underlay)
	m_primaryView   = new HistogramView(ui->transferPrimary,   true);
	m_secondaryView = new HistogramView(ui->transferSecondary, true);
	ui->gridLayout_transferPrimary->addWidget(  m_primaryView,   0, 0, 1, 1);
	ui->gridLayout_transferSecondary->addWidget(m_secondaryView, 0, 0, 1, 1);

	switchDimensions(1); // Default to 1D transfer function mode

    // Add the color selection widgets to the layout
    ui->layout_diffuseColor->addWidget(m_colorDiffuse);
    ui->layout_emissiveColor->addWidget(m_colorEmissive);
    ui->layout_specularColor->addWidget(m_colorSpecular);

    connect(m_colorDiffuse, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorDiffuseChanged(const QColor&)));
    connect(m_colorEmissive, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorEmissiveChanged(const QColor&)));
    connect(m_colorSpecular, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(colorSpecularChanged(const QColor&)));

    // Ensure transfer node selection is detected 
    connect(MainWindow::instance, SIGNAL(transferNodeSelected(std::shared_ptr<vox::Node>)),
        this, SLOT(setSelectedNode(std::shared_ptr<vox::Node>)));

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
//  Sets the selected node for transfer function editing
// ----------------------------------------------------------------------------
void TransferWidget::setSelectedNode(std::shared_ptr<vox::Node> node)
{
    m_currentNode = node;
    
    // :TODO: Prevent signal launches here
    // :TODO: Toggle selected state by update signal

    if (node)
    {
        auto material = m_currentNode->material();
        
        // Enable all off the active node controls
        ui->groupBox_currNode->setDisabled(false);
        ui->groupBox_nodePos->setDisabled(false);
        ui->groupBox_material->setDisabled(false);

        // Update all of the active node controls
        ui->doubleSpinBox_density->setValue(m_currentNode->position(0)*100.0f);
        ui->doubleSpinBox_gradient->setValue(m_currentNode->position(1)*100.0f);
        ui->doubleSpinBox_gradient2->setValue(m_currentNode->position(2)*100.0f);
        ui->doubleSpinBox_gloss->setValue(material->glossiness()*100.0f);
        ui->doubleSpinBox_opacity->setValue(material->opticalThickness()*100.0f);
        ui->checkBox_visible->setChecked(true);

        m_colorDiffuse->setColor(filescope::toQColor(material->diffuse()), true);
        m_colorEmissive->setColor(filescope::toQColor(material->emissive()), true);
        m_colorSpecular->setColor(filescope::toQColor(material->specular()), true);
    }
    else
    {
        // Disable all of the active node controls
        ui->groupBox_currNode->setDisabled(true);
        ui->groupBox_nodePos->setDisabled(true);
        ui->groupBox_material->setDisabled(true);
    }
}

// ----------------------------------------------------------------------------
//  Synchronizes the transfer function widget with the active
// ----------------------------------------------------------------------------
void TransferWidget::synchronizeView()
{
    // Update the local scene component handles
    m_transfer    = MainWindow::instance->scene().transfer;
    m_currentNode = m_transfer->nodes().empty() ? 
        nullptr : m_transfer->nodes().front();

    onTransferFunctionChanged();
}

// ----------------------------------------------------------------------------
//  Applies interactions with the interface to the scene transfer function
//   (Executed on the master renderer thread allowing safe scene changes)
// ----------------------------------------------------------------------------
void TransferWidget::processInteractions()
{
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
	msgBox.setIcon( QMessageBox::Question );
	msgBox.setText( tr("Are you sure you want to switch dimension?\
					Any unsaved changes to the current function will be lost.") );
	msgBox.setWindowTitle( "Switch Dimensions" );

	QPushButton* accept = msgBox.addButton( tr("Yes"), QMessageBox::AcceptRole );
	msgBox.addButton( tr("No"), QMessageBox::RejectRole );
	QPushButton* cancel = msgBox.addButton( tr("Cancel"), QMessageBox::RejectRole );
	msgBox.setDefaultButton( cancel );

	msgBox.exec( );

	if( msgBox.clickedButton( ) != accept ) return false;

	return true;
}

// ----------------------------------------------------------------------------
// Blocks radio button signals 
// ----------------------------------------------------------------------------
void TransferWidget::switchDimensions(int nDims)
{
	// Block radio button signals
	ui->radioButton_1->blockSignals( true );
	ui->radioButton_2->blockSignals( true );
	ui->radioButton_3->blockSignals( true );

	// Switch transfer control context
	switch( nDims ) 
	{
		case 1:

			// Configure widget for 1-Dimensional work 
            m_primaryView->setType(HistogramView::DataType_Density);
			ui->radioButton_1->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->hide( );
			ui->doubleSpinBox_gradient->hide( );
			ui->label_gradient->hide( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
			m_dimensions = 1;
			break;

		case 2:

			// Configure widget for 2-Dimensional work
            m_primaryView->setType(HistogramView::DataType_DensityGrad);
			ui->radioButton_2->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
			m_dimensions = 2;
			break;

		case 3:

			// Configure widget for 3-Dimensional work
            m_primaryView->setType(HistogramView::DataType_DensityGrad);
            m_secondaryView->setType(HistogramView::DataType_DensityLap);
			ui->radioButton_3->setChecked( true ); 
			ui->groupBox_transferSecondary->show( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->show( );
			ui->doubleSpinBox_gradient2->show( );
			ui->label_gradient2->show( );
			m_dimensions = 3;
			break; 

        default:
            break;
	}

	// Unblock radio button signals
	ui->radioButton_1->blockSignals( false );
	ui->radioButton_2->blockSignals( false );
	ui->radioButton_3->blockSignals( false );
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
// Select the first node / previous region
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_first_clicked( )
{
    if (m_transfer)
    {
        setSelectedNode(m_transfer->nodes().front());
    }
}

// ----------------------------------------------------------------------------
//  Select the next node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_next_clicked( )
{
    if (m_currentNode)
    {
        // Check if this is the last node
        auto nodes = m_transfer->nodes();
        if (m_currentNode != nodes.back())
        {
            // Acquire a handle to the subsequent node in the linked list
            auto iter = std::find(nodes.begin(), nodes.end(), m_currentNode);
            setSelectedNode(*(++iter));
        }
    }
}

// ----------------------------------------------------------------------------
//  Select the previous node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_prev_clicked( )
{
    if (m_currentNode)
    {
        // Check if this is the first node
        auto nodes = m_transfer->nodes();
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
void TransferWidget::on_pushButton_last_clicked( )
{
    if (m_transfer)
    {
        setSelectedNode(m_transfer->nodes().back());
    }
}

// ----------------------------------------------------------------------------
//  Delete selected node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_delete_clicked()
{
    if (m_currentNode)
    {
        m_transfer->removeNode(m_currentNode);
    }

    onTransferFunctionChanged();

    setSelectedNode(nullptr);

    //on_pushButton_first_clicked();
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

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_density->value() / 100.0f;
        m_currentNode->setPosition(0, value);
        emit nodePositionChanged(m_currentNode);
    }
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

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_density->value() / 100.0f;
        m_currentNode->setPosition(0, value);
        emit nodePositionChanged(m_currentNode);
    }
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

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_gloss->value() / 100.0f;
        m_currentNode->material()->setGlossiness(value);
    }
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
    
    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_gloss->value() / 100.0f;
        m_currentNode->material()->setGlossiness(value);
    }
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

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_opacity->value() / 100.0f;
        m_currentNode->material()->setOpticalThickness(value);
        emit nodePositionChanged(m_currentNode);
    }
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

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_opacity->value() / 100.0f;
        m_currentNode->material()->setOpticalThickness(value);
        emit nodePositionChanged(m_currentNode);
    }
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

    if (m_currentNode)
    {
        m_currentNode->material()->setEmissiveStrength(ui->doubleSpinBox_emissiveStr->value());
    }
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

    if (m_currentNode)
    {
        m_currentNode->material()->setEmissiveStrength(ui->doubleSpinBox_emissiveStr->value());
    }
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void TransferWidget::colorDiffuseChanged(QColor const& color)
{
    m_currentNode->material()->setDiffuse( Vector<UInt8,3>(color.red(), color.green(), color.blue()) );
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void TransferWidget::colorEmissiveChanged(QColor const& color)
{
    m_currentNode->material()->setEmissive( Vector<UInt8,3>(color.red(), color.green(), color.blue()) );
}

// --------------------------------------------------------------------
//  Signals a color change in the color selection widget
// --------------------------------------------------------------------
void TransferWidget::colorSpecularChanged(QColor const& color)
{
    m_currentNode->material()->setSpecular( Vector<UInt8,3>(color.red(), color.green(), color.blue()) );
}