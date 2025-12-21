# aws
provider "aws" {
  region = "eu-west-2" # London Region
}

# security group
resource "aws_security_group" "dashboard_sg" {
  name        = "dashboard-terraform-sg"
  description = "Allow SSH and Flask traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow ports
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 3. Create the Server (EC2)
resource "aws_instance" "app_server" {
  ami = "ami-0a0ff88d0f3f85a14"
  instance_type = "t3.micro"

  # aws key
  key_name      = "dashboard-key-pem"

  vpc_security_group_ids = [aws_security_group.dashboard_sg.id]

  # running docker
  user_data = <<-EOF
              #!/bin/bash
              # Update Linux
              apt-get update -y

              # Install Docker
              apt-get install -y docker.io

              # Start Docker
              systemctl start docker
              systemctl enable docker

              # Add 'ubuntu' user to docker group
              usermod -aG docker ubuntu
              EOF

  tags = {
    Name = "Volatility-Dashboard-Terraform"
  }
}

# Output the IP Address
output "server_public_ip" {
  value = aws_instance.app_server.public_ip
}